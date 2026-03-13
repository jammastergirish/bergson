"""
Replicate the GPT-2 WikiText experiment from the MAGIC paper (arXiv:2504.16430).

Setup from the paper:
  - Fine-tune pretrained GPT-2 125M on WikiText-2 (causal LM)
  - 4608 train chunks, 256 test chunks, each 512 tokens
  - 4 epochs of training
  - Adam: β1=0.95, β2=0.975, weight_decay=1e-5, ε_root=1e-8
  - LR: 0.0008, one-cycle linear schedule (peak at 25% of training)
  - MAGIC backward to get per-example attribution scores
  - LDS evaluation: drop 1% and 5% of training data, compute Spearman ρ

Usage:
  # Phase 1: Score (can parallelize across GPU groups)
  python examples/gpt2_wikitext.py score --save_dir /tmp/gpt2_wikitext_0 --test_start 0 --num_test_samples 25
  python examples/gpt2_wikitext.py score --save_dir /tmp/gpt2_wikitext_1 --test_start 25 --num_test_samples 25

  # Phase 2: Merge scores from parallel workers
  python examples/gpt2_wikitext.py merge --save_dir /tmp/gpt2_wikitext

  # Phase 3: Evaluate LDS (leave-k-out Spearman correlation)
  python examples/gpt2_wikitext.py lds --save_dir /tmp/gpt2_wikitext
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset, load_dataset
from simple_parsing import ArgumentParser
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from torchopt.typing import Numeric
from transformers import AutoTokenizer, GPT2LMHeadModel

from bergson.config import DistributedConfig
from bergson.distributed import (
    grad_tree,
    launch_distributed_run,
    shallow_copy,
    simple_fsdp,
)
from bergson.trainer import (
    BackwardState,
    DataStream,
    Trainer,
    TrainerState,
)
from bergson.utils.math import weighted_causal_lm_ce


@dataclass
class RunConfig:
    save_dir: str = "/tmp/gpt2_wikitext"
    """Directory for checkpoints and results."""

    num_train_samples: int = 4608
    """Number of train chunks to use (paper: 4608)."""

    batch_size: int = 8
    """Global batch size."""

    max_length: int = 512
    """Context length for tokenized chunks."""

    num_epochs: int = 4
    """Number of training epochs."""

    lr: float = 0.0008
    """Peak learning rate."""

    num_test_samples: int = 256
    """Number of test samples to compute attribution for (paper: 256)."""

    test_start: int = 0
    """First test sample index (for parallelizing across GPU groups)."""

    num_subsets: int = 100
    """Number of random subsets for LDS evaluation."""

    subset_start: int = 0
    """First subset index (for parallelizing LDS across GPUs)."""

    drop_fractions: list[float] = field(default_factory=lambda: [0.01, 0.05])
    """Drop fractions for LDS evaluation."""

    seed: int = 42
    """Random seed."""

    eps_root: float = 1e-2
    """eps_root for AdamW (goes inside the sqrt in the denominator)."""


def prepare_wikitext(max_length: int = 512) -> tuple[Dataset, Dataset]:
    """Load WikiText-2 and chunk into fixed-length token sequences."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    chunks = {}
    for split_name, target_key in [("train", "train"), ("test", "test")]:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split_name)
        assert isinstance(ds, Dataset)

        # Concatenate all text into one long string
        all_text = "\n".join([t for t in ds["text"] if t.strip()])
        tokens = tokenizer(all_text, return_tensors="pt")["input_ids"].squeeze(0)

        # Chunk into fixed-length sequences
        n_chunks = len(tokens) // max_length
        tokens = tokens[: n_chunks * max_length].view(n_chunks, max_length)

        # Convert to HF Dataset
        chunks[target_key] = Dataset.from_dict({"input_ids": tokens.tolist()})

    print(f"WikiText-2 chunks: train={len(chunks['train'])}, test={len(chunks['test'])}")
    return chunks["train"], chunks["test"]


class ChunkedDataStream(DataStream):
    """DataStream for pre-tokenized chunks (input_ids already in dataset)."""

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= self.num_batches:
            raise IndexError("DataStream index out of range")

        indices = list(
            range(
                i * self.batch_size + self.rank,
                (i + 1) * self.batch_size,
                self.world_size,
            )
        )
        raw = self.dataset[indices]

        input_ids = torch.tensor(raw["input_ids"], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "example_weight": self.weights[
                i * self.batch_size
                + self.rank : (i + 1) * self.batch_size : self.world_size
            ],
        }


def make_model(rank: int, world_size: int = 1) -> GPT2LMHeadModel:
    """Load pretrained GPT-2 125M."""
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.set_attn_implementation("eager")
    model.loss_function = weighted_causal_lm_ce

    # Untie lm_head from wte so shallow_copy doesn't drop one of the tied params
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())

    # Resize embeddings so vocab is divisible by world_size (FSDP Shard(0) pads
    # if not, causing a vocab_size mismatch in the loss function).
    if model.config.vocab_size % world_size != 0:
        new_size = ((model.config.vocab_size // world_size) + 1) * world_size
        model.resize_token_embeddings(new_size)

    model.to(f"cuda:{rank}")
    return model


def make_optimizer(run_cfg: RunConfig, num_steps: int):
    """Paper schedule: start at 1e-6×peak, warm up over 25%, then decay to 0.1×peak."""
    warmup_steps = int(0.25 * num_steps)
    start_lr = run_cfg.lr * 1e-6
    end_lr = run_cfg.lr * 0.1

    def schedule(step: Numeric) -> Numeric:
        if step < warmup_steps:
            return start_lr + (run_cfg.lr - start_lr) * step / max(1, warmup_steps)
        # Linear decay from peak to 0.1×peak.
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return run_cfg.lr - (run_cfg.lr - end_lr) * progress

    return torchopt.adamw(
        schedule,
        betas=(0.95, 0.975),
        eps_root=run_cfg.eps_root,
        weight_decay=1e-5,
    )


def _cleanup_intermediate_ckpts(ckpt_dir: str, original_names: set[str]):
    """Delete checkpoint files that weren't in the original forward pass set."""
    for name in os.listdir(ckpt_dir):
        if name not in original_names:
            path = os.path.join(ckpt_dir, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def _clone_state_tensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clone a parameter/buffer dict, making independent copies of all data."""
    return {k: v.detach().clone().requires_grad_(False) for k, v in tensors.items()}


# ─── Phase 1: Score ──────────────────────────────────────────────────────────


def score_worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_ds: Dataset,
    test_ds: Dataset,
    run_cfg: RunConfig,
):
    torch.manual_seed(run_cfg.seed)
    torch.cuda.set_device(rank)

    model = make_model(rank, world_size)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=4),
            world_size=world_size,
        )
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            model = simple_fsdp(model)

    # Match the paper's 4608-train-chunk setup, while still requiring a whole batch.
    assert run_cfg.num_train_samples <= len(train_ds)

    # TODO This will drop data that doesn't evenly divide inot batch
    n_train = (run_cfg.num_train_samples // run_cfg.batch_size) * run_cfg.batch_size
    train_ds = train_ds.select(range(n_train))

    # 4 epochs: repeat the dataset
    epoch_indices = list(range(n_train)) * run_cfg.num_epochs
    full_train_ds = train_ds.select(epoch_indices)

    num_steps = len(full_train_ds) // run_cfg.batch_size
    opt = make_optimizer(run_cfg, num_steps)
    trainer, fwd_state = Trainer.initialize(model, opt)

    ckpt_dir = os.path.join(run_cfg.save_dir, "ckpts")

    train_stream = ChunkedDataStream(
        full_train_ds,
        processor=None,
        batch_size=run_cfg.batch_size,
        device=f"cuda:{rank}",
        max_length=run_cfg.max_length,
    )

    if global_rank == 0:
        print(f"Training: {n_train} examples × {run_cfg.num_epochs} epochs = "
              f"{len(full_train_ds)} total, {num_steps} steps, batch_size={run_cfg.batch_size}")

    # ── Forward pass with checkpointing ──
    fwd_state = trainer.train(
        fwd_state, train_stream, inplace=True, save_dir=ckpt_dir
    )

    # Save the final state so we can restore it for each test sample
    final_state_path = os.path.join(run_cfg.save_dir, "final_state.ckpt")
    fwd_state.save(final_state_path).result()

    sqrt_ckpt_names = set(os.listdir(ckpt_dir))
    if global_rank == 0:
        print(f"Forward pass done. {len(sqrt_ckpt_names)} sqrt checkpoints saved.")

    # ── Backward pass for each test sample ──
    test_end = min(run_cfg.test_start + run_cfg.num_test_samples, len(test_ds))
    test_range = range(run_cfg.test_start, test_end)
    n_test = len(test_range)

    print(n_test, "test samples.")

    # Resume from previously saved partial scores if they exist
    scores_path = os.path.join(run_cfg.save_dir, "scores.pt")
    if global_rank == 0 and os.path.exists(scores_path):
        all_scores = list(torch.load(scores_path, weights_only=True).unbind(0))
        print(f"Resuming: loaded {len(all_scores)} previously scored samples")
    else:
        all_scores = []

    for t_idx in test_range:
        sample_num = t_idx - run_cfg.test_start
        # Skip already-scored samples (resume support)
        if sample_num < len(all_scores):
            if global_rank == 0:
                print(f"Skipping test sample {t_idx} (already scored)")
            continue

        if global_rank == 0:
            print(f"\nScoring test sample {t_idx+1}/{test_end} "
                  f"(#{sample_num+1}/{n_test} in this worker)")

        if world_size > 1:
            dist.barrier()

        # Restore the final trained state
        fwd_state.detach_()
        fwd_state.load(final_state_path)

        # Fresh data stream (clean weights for this backward pass)
        train_stream = ChunkedDataStream(
            full_train_ds,
            processor=None,
            batch_size=run_cfg.batch_size,
            device=f"cuda:{rank}",
            max_length=run_cfg.max_length,
        )

        # Load a single test sample onto this rank
        test_ex = test_ds[t_idx]
        input_ids = torch.tensor([test_ex["input_ids"]], device=f"cuda:{rank}")

        # Construct backward state from this test sample's loss
        with fwd_state.activate(model) as params:
            loss = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone(),
            ).loss
            grads = grad_tree(loss, params, create_graph=True)
            opt_grads = [
                torch.zeros_like(buf)
                for buf in tree_iter(fwd_state.opt_state)
                if isinstance(buf, torch.Tensor) and buf.is_floating_point()
            ]
            bwd_state = BackwardState(
                grads, opt_grads, torch.zeros_like(train_stream.weights)
            )

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        if global_rank == 0:
            print(f"  Test loss: {loss.item():.4f}")

        # Preserve the original sqrt checkpoints only when we need them for
        # additional test samples. A single-sample score run can safely clean
        # replay checkpoints as it goes to avoid multi-terabyte growth.
        bwd_state = trainer.backward(
            ckpt_dir, train_stream, bwd_state, fwd_state, inplace=True,
            cleanup=n_test == 1,
        )

        # Remove intermediate checkpoints created during backward replay,
        # keeping only the original sqrt checkpoints for the next test sample.
        if global_rank == 0:
            _cleanup_intermediate_ckpts(ckpt_dir, sqrt_ckpt_names)

        if world_size > 1:
            dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.AVG)

        scores = bwd_state.weight_grads.detach().cpu()
        all_scores.append(scores)

        if global_rank == 0:
            nz = (scores != 0).sum().item()
            print(f"  Non-zero scores: {nz}/{len(scores)}, "
                  f"mean={scores.mean():.6f}, std={scores.std():.6f}")

            # Save incrementally after each test sample
            torch.save(torch.stack(all_scores), scores_path)

        if world_size > 1:
            dist.barrier()

    # ── Save config ──
    if global_rank == 0:
        scores_tensor = torch.stack(all_scores)
        torch.save(scores_tensor, scores_path)
        print(f"\nSaved scores tensor of shape {scores_tensor.shape} to {scores_path}")

        cfg_path = os.path.join(run_cfg.save_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "n_train": n_train,
                "n_test": n_test,
                "test_start": run_cfg.test_start,
                "num_epochs": run_cfg.num_epochs,
                "num_steps": num_steps,
                "batch_size": run_cfg.batch_size,
                "max_length": run_cfg.max_length,
                "lr": run_cfg.lr,
            }, f, indent=2)

    # Final cleanup
    if global_rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(final_state_path, ignore_errors=True)


# ─── Phase 2: LDS Evaluation ─────────────────────────────────────────────────


def lds_worker(
    global_rank: int,
    rank: int,
    world_size: int,
    train_ds: Dataset,
    test_ds: Dataset,
    run_cfg: RunConfig,
):
    """Retrain with random subsets dropped and compute Spearman correlation."""
    from scipy.stats import spearmanr

    torch.cuda.set_device(rank)

    # Load scores
    scores_path = os.path.join(run_cfg.save_dir, "scores.pt")
    scores = torch.load(scores_path, weights_only=True)  # [n_test, n_train * n_epochs]
    n_test = scores.shape[0]

    cfg_path = os.path.join(run_cfg.save_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_train = cfg["n_train"]
    num_steps = cfg["num_steps"]
    num_epochs = cfg.get("num_epochs", run_cfg.num_epochs)

    model = make_model(rank, world_size)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=4),
            world_size=world_size,
        )
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            model = simple_fsdp(model)

    train_ds = train_ds.select(range(n_train))
    epoch_indices = list(range(n_train)) * num_epochs
    full_train_ds = train_ds.select(epoch_indices)

    # Save pretrained params before Trainer moves them to meta device.
    # We need these to re-initialize the state for each retrain.
    pretrained_params = _clone_state_tensors(
        {
            k: v
            for k, v in model.named_parameters(remove_duplicate=False)
            if v.requires_grad
        }
    )
    pretrained_buffers = _clone_state_tensors(
        dict(model.named_buffers(remove_duplicate=False))
    )

    opt = make_optimizer(run_cfg, num_steps)
    trainer = Trainer(model, opt)

    n_test_eval = min(run_cfg.num_test_samples, n_test, len(test_ds))

    subset_end = run_cfg.subset_start + run_cfg.num_subsets

    for drop_frac in run_cfg.drop_fractions:
        n_drop = int(n_train * drop_frac)
        if global_rank == 0:
            print(f"\n{'='*60}")
            print(f"  LDS evaluation: drop {drop_frac:.0%} ({n_drop} examples)")
            print(f"  Subsets {run_cfg.subset_start}..{subset_end}")
            print(f"{'='*60}")

        # Use a fixed generator to produce deterministic permutations.
        # Fast-forward to subset_start so different workers get different subsets
        # but the same subset index always yields the same permutation.
        gen = torch.Generator().manual_seed(run_cfg.seed)
        for _ in range(run_cfg.subset_start):
            torch.randperm(n_train, generator=gen)

        # Per-test-sample: list of (predicted_losses, true_losses) across subsets
        all_predicted = [[] for _ in range(n_test_eval)]
        all_true = [[] for _ in range(n_test_eval)]

        for s_idx in range(run_cfg.subset_start, subset_end):
            torch.manual_seed(run_cfg.seed)

            # Random subset to drop
            perm = torch.randperm(n_train, generator=gen)
            drop_indices = set(perm[:n_drop].tolist())

            # Build weight vector: 0 for dropped, 1 for kept
            weights = torch.ones(len(full_train_ds))
            for epoch in range(num_epochs):
                for idx in drop_indices:
                    weights[epoch * n_train + idx] = 0.0

            # Predicted loss change from MAGIC scores (linear approximation)
            drop_list = sorted(drop_indices)
            drop_positions = torch.tensor(
                [epoch * n_train + idx
                 for epoch in range(num_epochs)
                 for idx in drop_list]
            )
            preds = -scores[:n_test_eval].index_select(1, drop_positions).sum(dim=1)
            for t_idx in range(n_test_eval):
                all_predicted[t_idx].append(preds[t_idx].item())

            # Fresh state from saved pretrained weights
            params = _clone_state_tensors(pretrained_params)
            opt_state = trainer.optimizer.init(params)
            buffers = _clone_state_tensors(pretrained_buffers)
            state = TrainerState(params, opt_state, buffers)

            train_stream = ChunkedDataStream(
                full_train_ds,
                processor=None,
                batch_size=run_cfg.batch_size,
                device=f"cuda:{rank}",
                max_length=run_cfg.max_length,
            )
            train_stream.weights.data.copy_(weights.to(train_stream.weights.device))

            state = trainer.train(state, train_stream, inplace=True)

            # Evaluate on each test sample
            with torch.no_grad(), state.activate(model):
                for t_idx in range(n_test_eval):
                    test_ex = test_ds[t_idx]
                    input_ids = torch.tensor(
                        [test_ex["input_ids"]], device=f"cuda:{rank}"
                    )
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        labels=input_ids.clone(),
                    ).loss
                    if world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    all_true[t_idx].append(loss.item())

            if global_rank == 0:
                local_idx = s_idx - run_cfg.subset_start + 1
                if len(all_true[0]) >= 3:
                    rho = spearmanr(all_predicted[0], all_true[0]).statistic
                    print(f"  Subset {s_idx+1} ({local_idx}/{run_cfg.num_subsets}): "
                          f"ρ(test #0) = {rho:.4f}")
                else:
                    print(f"  Subset {s_idx+1} ({local_idx}/{run_cfg.num_subsets}): computing...")

        # Save partial results (will be merged later if parallelized)
        if global_rank == 0:
            results_path = os.path.join(
                run_cfg.save_dir,
                f"lds_drop{int(drop_frac*100)}_s{run_cfg.subset_start}.json",
            )
            with open(results_path, "w") as f:
                json.dump({
                    "drop_fraction": drop_frac,
                    "n_drop": n_drop,
                    "subset_start": run_cfg.subset_start,
                    "subset_end": subset_end,
                    "n_subsets": run_cfg.num_subsets,
                    "predicted": [p for p in all_predicted],
                    "true": [t for t in all_true],
                }, f, indent=2)
            print(f"  Saved partial results to {results_path}")


# ─── Phase: Merge scores from parallel workers ───────────────────────────────


def merge_scores(run_cfg: RunConfig):
    """Merge scores.pt from parallel worker directories into a single file."""
    import glob as glob_mod

    worker_dirs = sorted(glob_mod.glob(run_cfg.save_dir + "_*"))
    if not worker_dirs:
        print(f"No worker directories found matching {run_cfg.save_dir}_*")
        return

    all_scores = []
    base_cfg = None
    for d in worker_dirs:
        scores_path = os.path.join(d, "scores.pt")
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(scores_path):
            print(f"  Warning: {scores_path} not found, skipping")
            continue

        s = torch.load(scores_path, weights_only=True)
        all_scores.append(s)
        print(f"  {d}: {s.shape[0]} test samples, shape {s.shape}")

        if base_cfg is None and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                base_cfg = json.load(f)

    if not all_scores:
        print("No scores found to merge.")
        return

    merged = torch.cat(all_scores, dim=0)
    os.makedirs(run_cfg.save_dir, exist_ok=True)

    scores_path = os.path.join(run_cfg.save_dir, "scores.pt")
    torch.save(merged, scores_path)
    print(f"\nMerged {len(all_scores)} workers -> {merged.shape} saved to {scores_path}")

    # Write config with correct total n_test
    if base_cfg is not None:
        base_cfg["n_test"] = merged.shape[0]
        base_cfg.pop("test_start", None)
        cfg_path = os.path.join(run_cfg.save_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(base_cfg, f, indent=2)
        print(f"Config saved to {cfg_path}")


# ─── Phase: Merge LDS results from parallel workers ──────────────────────────


def merge_lds(run_cfg: RunConfig):
    """Merge partial LDS results and compute final Spearman correlations."""
    import glob as glob_mod
    from scipy.stats import spearmanr

    for drop_frac in run_cfg.drop_fractions:
        prefix = f"lds_drop{int(drop_frac*100)}_s"
        files = sorted(glob_mod.glob(os.path.join(run_cfg.save_dir, f"{prefix}*.json")))
        if not files:
            print(f"No partial LDS files found for drop {drop_frac:.0%}")
            continue

        # Merge predicted and true across all partial files
        all_predicted = None
        all_true = None
        total_subsets = 0
        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)
            if all_predicted is None:
                all_predicted = [list(p) for p in data["predicted"]]
                all_true = [list(t) for t in data["true"]]
            else:
                for i in range(len(all_predicted)):
                    all_predicted[i].extend(data["predicted"][i])
                    all_true[i].extend(data["true"][i])
            total_subsets += data["n_subsets"]
            print(f"  {fpath}: subsets {data['subset_start']}..{data['subset_end']}")

        n_test_eval = len(all_predicted)
        correlations = []
        for t_idx in range(n_test_eval):
            rho = spearmanr(all_predicted[t_idx], all_true[t_idx]).statistic
            correlations.append(rho)

        avg_lds = sum(correlations) / len(correlations)
        print(f"\n  Drop {drop_frac:.0%} LDS (avg Spearman over {total_subsets} subsets): {avg_lds:.4f}")
        print(f"  Per-sample ρ: {['%.3f' % c for c in correlations]}")

        cfg_path = os.path.join(run_cfg.save_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            n_drop = int(cfg["n_train"] * drop_frac)
        else:
            n_drop = 0

        results_path = os.path.join(run_cfg.save_dir, f"lds_drop{int(drop_frac*100)}.json")
        with open(results_path, "w") as f:
            json.dump({
                "drop_fraction": drop_frac,
                "n_drop": n_drop,
                "n_subsets": total_subsets,
                "avg_lds": avg_lds,
                "per_sample_rho": correlations,
            }, f, indent=2)
        print(f"  Final results saved to {results_path}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "phase", choices=["score", "merge", "lds", "merge_lds"],
        help="Which phase to run: score, merge, lds, or merge_lds",
    )
    parser.add_arguments(RunConfig, dest="run_cfg")
    parser.add_arguments(DistributedConfig, dest="dist_cfg")
    args = parser.parse_args()

    run_cfg: RunConfig = args.run_cfg
    dist_cfg: DistributedConfig = args.dist_cfg

    os.makedirs(run_cfg.save_dir, exist_ok=True)

    if args.phase == "merge":
        merge_scores(run_cfg)
        return
    elif args.phase == "merge_lds":
        merge_lds(run_cfg)
        return

    print("Preparing WikiText-2 data...")
    train_ds, test_ds = prepare_wikitext(run_cfg.max_length)

    if args.phase == "score":
        launch_distributed_run(
            "gpt2_score",
            score_worker,
            [train_ds, test_ds, run_cfg],
            dist_cfg,
        )
    elif args.phase == "lds":
        launch_distributed_run(
            "gpt2_lds",
            lds_worker,
            [train_ds, test_ds, run_cfg],
            dist_cfg,
        )


if __name__ == "__main__":
    main()
