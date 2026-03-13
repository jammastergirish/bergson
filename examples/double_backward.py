import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torchopt
from datasets import load_dataset
from scipy.stats import spearmanr
from simple_parsing import ArgumentParser
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from torchopt.typing import Numeric
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

from bergson.config import DistributedConfig
from bergson.distributed import grad_tree, launch_distributed_run, simple_fsdp
from bergson.trainer import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce


@dataclass
class RunConfig:
    model_name: str = "EleutherAI/pythia-160m"
    """HuggingFace model name."""

    dataset_name: str = "EleutherAI/SmolLM2-135M-10B"
    """HuggingFace dataset name."""

    dataset_split: str = "train"
    """Dataset split to use."""

    grad_checkpointing: bool = False
    """Whether to use gradient checkpointing during the forward pass."""

    lr: float = 1e-5
    """Base learning rate after warmup."""

    warmup_steps: int = 10
    """Number of warmup steps before applying base lr."""

    batch_size: int = 8
    """Per-device batch size."""

    num_batches: int = 25
    """Number of training batches."""

    max_length: int = 256
    """Maximum token sequence length."""

    save_dir: str = "/mnt/ssd-1/will-magic/magic-ckpts"
    """Directory to save forward pass checkpoints."""

    num_subsets: int = 100
    """Number of leave-one-out subsets for Spearman correlation."""

    seed: int = 42
    """Random seed for subset permutation."""


def worker(global_rank: int, rank: int, world_size: int, dataset, run_cfg: RunConfig):
    torch.cuda.set_device(rank)

    cfg = GPTNeoXConfig.from_pretrained(run_cfg.model_name, revision="step0")
    model = GPTNeoXForCausalLM(cfg)
    model.set_attn_implementation("eager")
    model.loss_function = weighted_causal_lm_ce
    model.to(f"cuda:{rank}")
    if run_cfg.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=dict(use_reentrant=False),
        )

    processor = AutoTokenizer.from_pretrained(run_cfg.model_name)
    processor.pad_token = processor.eos_token

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            model = simple_fsdp(model)

    def schedule(step: Numeric) -> Numeric:
        if step < run_cfg.warmup_steps:
            return 0.0
        return run_cfg.lr

    opt = torchopt.adamw(
        schedule,
        betas=(0.95, 0.975),
        eps_root=1e-8,
    )
    trainer, fwd_state = Trainer.initialize(model, opt)

    # save state0
    path0 = os.path.join(run_cfg.save_dir, "state0.pt")
    save_fut = fwd_state.save(path0)

    stream = DataStream(
        dataset,
        processor,
        batch_size=run_cfg.batch_size,
        num_batches=run_cfg.num_batches,
        device=f"cuda:{rank}",
        max_length=run_cfg.max_length,
    )
    fwd_state = trainer.train(
        fwd_state,
        stream,
        inplace=True,
        save_dir=run_cfg.save_dir,
    )

    with fwd_state.activate(model) as params:
        stream.requires_grad = True

        ex = stream[0]
        del ex["example_weight"]

        loss = model(**ex).loss

        grads = grad_tree(loss, params, create_graph=True)
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(fwd_state.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        bwd_state = BackwardState(grads, opt_grads, torch.zeros_like(stream.weights))

    if world_size > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    bwd_state = trainer.backward(
        run_cfg.save_dir,
        stream,
        bwd_state,
        fwd_state,
        inplace=True,
    )
    if world_size > 1:
        dist.all_reduce(bwd_state.weight_grads, op=dist.ReduceOp.AVG)
    if global_rank == 0:
        print(f"Scores 2: {bwd_state.weight_grads.tolist()}")

    baseline = loss.item()
    if global_rank == 0:
        print(f"Baseline: {baseline}")
        print("Grad:", bwd_state.weight_grads.sum())

    stream.requires_grad = False

    diffs = []
    score_sums = []

    gen = torch.Generator().manual_seed(run_cfg.seed)
    perm = torch.randperm(len(stream.weights), generator=gen)
    subsets = perm.chunk(run_cfg.num_subsets)

    save_fut.result()  # ensure state0 is saved before loading in loop
    fwd_state.load(path0)

    for subset in subsets:
        stream.weights.fill_(1.0)
        stream.weights[subset] = 0.0

        for x in stream:
            fwd_state = trainer.step(fwd_state, x)

        with fwd_state.activate(model):
            loss = model(**stream[0]).loss

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        diffs.append(baseline - loss.item())
        score_sums.append(bwd_state.weight_grads[subset].sum().item())

        corr = spearmanr(diffs, score_sums)
        if global_rank == 0:
            print(f"Loss diff: {diffs[-1]}")
            print(f"Score: {score_sums[-1]}")
            print(f"Spearman correlation: {corr}")


def main():
    parser = ArgumentParser()
    parser.add_arguments(RunConfig, dest="run_cfg")
    parser.add_arguments(DistributedConfig, dest="dist_cfg")
    args = parser.parse_args()

    run_cfg: RunConfig = args.run_cfg
    dist_cfg: DistributedConfig = args.dist_cfg

    ds = load_dataset(run_cfg.dataset_name, split=run_cfg.dataset_split)
    launch_distributed_run("double_backward", worker, [ds, run_cfg], dist_cfg)


if __name__ == "__main__":
    main()
