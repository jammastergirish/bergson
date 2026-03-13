"""
Sweep sequence length to see how MAGIC vs FD correlation degrades.
Tests 64, 128, 256, 512 tokens with GPT-2/WikiText.
"""

import gc
import os
import shutil
import tempfile

import torch
import torchopt
from datasets import Dataset, load_dataset
from scipy.stats import spearmanr
from torchopt.pytree import tree_iter
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

from bergson.distributed import grad_tree
from bergson.trainer import BackwardState, DataStream, Trainer, TrainerState
from bergson.utils.math import weighted_causal_lm_ce
from bergson.chunk_and_tokenize import chunk_and_tokenize

# Disable autocast
torch.cuda.is_bf16_supported = lambda *a, **k: False


class ChunkedDataStream(DataStream):
    def __getitem__(self, i):
        if i < 0 or i >= self.num_batches:
            raise IndexError()
        indices = list(range(
            i * self.batch_size + self.rank,
            (i + 1) * self.batch_size,
            self.world_size,
        ))
        raw = self.dataset[indices]
        input_ids = raw["input_ids"].detach().clone().to(self.device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
            "example_weight": self.weights[
                i * self.batch_size + self.rank:(i + 1) * self.batch_size:self.world_size
            ],
        }


def make_model(device):
    config = GPT2Config.from_pretrained("gpt2")
    config.attn_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.resid_pdrop = 0.0
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config, torch_dtype=torch.float32)
    model.set_attn_implementation("eager")
    model.loss_function = weighted_causal_lm_ce
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
    model.to(device)
    return model


def run_test(max_length, train_ds, test_ids, batch_size, device):
    n_train = len(train_ds)
    input_ids = torch.tensor([test_ids], device=device)

    # Save pretrained params for FD
    model_ref = make_model(device)
    pp = {k: v.detach().clone() for k, v in model_ref.named_parameters(remove_duplicate=False) if v.requires_grad}
    pb = {k: v.detach().clone() for k, v in model_ref.named_buffers(remove_duplicate=False)}
    del model_ref

    # MAGIC
    model = make_model(device)
    torch.manual_seed(42)
    opt = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2, weight_decay=1e-5)
    trainer, fwd = Trainer.initialize(model, opt)
    ckpt = tempfile.mkdtemp()
    stream = ChunkedDataStream(train_ds, processor=None, batch_size=batch_size, device=device, max_length=max_length)
    fwd = trainer.train(fwd, stream, inplace=True, save_dir=ckpt)
    fwd.save(os.path.join(ckpt, "final_state.ckpt")).result()

    stream2 = ChunkedDataStream(train_ds, processor=None, batch_size=batch_size, device=device, max_length=max_length)
    with fwd.activate(model) as params:
        test_loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids.clone()).loss
        grads = grad_tree(test_loss, params, create_graph=True)
        opt_grads = [torch.zeros_like(buf) for buf in tree_iter(fwd.opt_state) if isinstance(buf, torch.Tensor) and buf.is_floating_point()]
        bwd = BackwardState(grads, opt_grads, torch.zeros_like(stream2.weights))

    bwd = trainer.backward(ckpt, stream2, bwd, fwd, inplace=True, cleanup=True)
    scores = bwd.weight_grads.detach().cpu()
    shutil.rmtree(ckpt, ignore_errors=True)
    del model, trainer, fwd, bwd, grads, opt_grads, test_loss
    gc.collect()
    torch.cuda.synchronize()

    # FD
    eps = 1e-2
    fd_vals = []
    for ex_idx in range(n_train):
        losses = {}
        for sign, w in [("plus", 1.0 + eps), ("minus", 1.0 - eps)]:
            torch.manual_seed(42)
            model_fd = make_model(device)
            p = {k: v.detach().clone().requires_grad_(False) for k, v in pp.items()}
            o = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2, weight_decay=1e-5)
            t = Trainer(model_fd, o)
            s = TrainerState(p, o.init(p), {k: v.detach().clone() for k, v in pb.items()})
            st = ChunkedDataStream(train_ds, processor=None, batch_size=batch_size, device=device, max_length=max_length)
            st.weights.data[ex_idx] = w
            s = t.train(s, st, inplace=True)
            with torch.no_grad(), s.activate(model_fd):
                l = model_fd(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids.clone()).loss.item()
            losses[sign] = l
            del model_fd, t, s, st
        fd_vals.append((losses["plus"] - losses["minus"]) / (2 * eps))

    rho = spearmanr(scores.tolist(), fd_vals).statistic

    ratios = []
    for i in range(n_train):
        ratio = scores[i].item() / fd_vals[i] if abs(fd_vals[i]) > 1e-12 else float('inf')
        ratios.append(ratio)

    mean_ratio = sum(ratios) / len(ratios)
    print(f"  {max_length:4d} tok:  Spearman={rho:.4f}  mean_ratio={mean_ratio:.3f}  "
          f"ratio_range=[{min(ratios):.3f}, {max(ratios):.3f}]")

    del pp, pb
    gc.collect()
    torch.cuda.synchronize()
    return rho, mean_ratio, ratios


def main():
    device = "cuda:0"
    torch.cuda.set_device(0)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    n_train = 16
    batch_size = 8

    results = {}
    for max_length in [64, 128, 256, 512]:
        ds = chunk_and_tokenize(ds, tokenizer, max_seq_len=max_length)
        tokens = ds["input_ids"][:]
        n_train = len(tokens)

        print(n_train)

        train_ds = ds.select(range(len(ds) - 1))

        test_ids = tokens[n_train - 1].tolist()
        print(test_ids, type(train_ds))

        rho, mean_ratio, ratios = run_test(max_length, train_ds, test_ids, batch_size, device)
        results[max_length] = (rho, mean_ratio)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for ml, (rho, mr) in results.items():
        print(f"  {ml:4d} tok:  Spearman={rho:.4f}  mean_ratio={mr:.3f}")


if __name__ == "__main__":
    main()
