"""MAGIC integration test: forward + backward through 2 training steps."""

import tempfile

import pytest
import torch
import torchopt
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.utils.math import weighted_causal_lm_ce

MODEL_CONFIGS = [
    "trl-internal-testing/tiny-Phi3ForCausalLM",
    "EleutherAI/pythia-14m",
]


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_two_steps(model_name, dataset):
    device = "cpu"

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )

    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    train_stream = DataStream(
        dataset,
        batch_size=len(dataset),
        device=device,
    )
    assert len(train_stream) == 1

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=True,
            save_dir=ckpt_dir,
        )

        # Compute query gradients on the training batch
        with fwd_state.activate(model) as params:
            batch = train_stream[0]
            del batch["example_weight"]
            loss = model(**batch).loss
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }

            opt_grads = [
                torch.zeros_like(buf)
                for buf in tree_iter(fwd_state.opt_state)
                if isinstance(buf, torch.Tensor) and buf.is_floating_point()
            ]
            bwd_state = BackwardState(
                query_grads,
                opt_grads,
                torch.zeros_like(train_stream.weights),
            )

        # Backward pass through training
        train_stream.requires_grad = True
        bwd_state = trainer.backward(
            ckpt_dir,
            train_stream,
            bwd_state,
            fwd_state,
            inplace=True,
            cleanup=True,
        )

    scores = bwd_state.weight_grads.detach().cpu()
    assert scores.shape == (len(dataset),)
    assert scores.abs().sum() > 0, "Attribution scores are all zero"


def _run_magic(model_name, dataset, weight_shape, seed=42, device="cpu"):
    """Run a 1-step MAGIC cycle and return the weight_grads (MAGIC scores)."""
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    train_stream = DataStream(
        dataset,
        batch_size=len(dataset),
        device=device,
        weight_shape=weight_shape,
    )

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(
            fwd_state, train_stream, inplace=True, save_dir=ckpt_dir
        )

        with fwd_state.activate(model) as params:
            batch = train_stream[0]
            del batch["example_weight"]
            loss = model(**batch).loss
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }
            opt_grads = [
                torch.zeros_like(buf)
                for buf in tree_iter(fwd_state.opt_state)
                if isinstance(buf, torch.Tensor) and buf.is_floating_point()
            ]
            bwd_state = BackwardState(
                query_grads, opt_grads, torch.zeros_like(train_stream.weights)
            )

        train_stream.requires_grad = True
        bwd_state = trainer.backward(
            ckpt_dir,
            train_stream,
            bwd_state,
            fwd_state,
            inplace=True,
            cleanup=True,
        )

    return bwd_state.weight_grads.detach().cpu()


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_per_token_scores_zero_at_masked_labels(model_name):
    """MAGIC scores are exactly zero at positions whose weight has no loss path.

    Two sources of zero-by-construction in weighted_causal_lm_ce:
    - shifted labels == -100: F.cross_entropy with ignore_index=-100 makes
      tok_loss[t] == 0, so w[:, t] enters the loss multiplied by zero.
    - the last-token weight slot: example_weight is sliced as [:, :-1] before
      multiplication, so column T-1 never enters the loss.
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "labels": [[1, 2, -100, 4, 5], [-100, 7, 8, -100, 10]],
            "attention_mask": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        }
    )
    N, T = len(ds), 5

    per_tok = _run_magic(model_name, ds, weight_shape=(N, T))
    assert per_tok.shape == (N, T)

    labels = torch.tensor(ds["labels"])
    zero_mask = torch.zeros(N, T, dtype=torch.bool)
    zero_mask[:, T - 1] = True  # unused last-token slot
    zero_mask[:, :-1] = labels[:, 1:] == -100  # shifted masked positions

    assert torch.all(per_tok[zero_mask] == 0), (
        f"Expected zero MAGIC scores at masked/unused positions; "
        f"got max |score| = {per_tok[zero_mask].abs().max():.3e}"
    )
    assert per_tok[~zero_mask].abs().sum() > 0, (
        "All non-masked positions are zero — test is degenerate"
    )


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_per_token_sums_to_per_doc(model_name, dataset):
    """Per-token MAGIC scores summed over tokens equal per-doc MAGIC scores.

    MAGIC computes d(query_loss)/dw through the training trajectory. With
    weighted_causal_lm_ce, the training loss is
        per-doc:   sum_{i,t} w_i     * tok_loss[i,t] / denom
        per-token: sum_{i,t} w_{i,t} * tok_loss[i,t] / denom
    Both evaluate to the same value at initialization (all weights = 1), so the
    two runs share an identical training trajectory. By linearity of the MAGIC
    backward pass, dQ/dw_i = sum_t dQ/dw_{i,t}.
    """
    N = len(dataset)
    T = len(dataset[0]["input_ids"])

    per_doc = _run_magic(model_name, dataset, weight_shape=(N,))
    per_tok = _run_magic(model_name, dataset, weight_shape=(N, T))

    assert per_doc.shape == (N,)
    assert per_tok.shape == (N, T)

    torch.testing.assert_close(per_tok.sum(dim=-1), per_doc, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("model_name", MODEL_CONFIGS)
def test_magic_per_token_sums_to_per_doc_packed(model_name):
    """Per-doc MAGIC (1D weights via doc_ids lookup) equals per-token MAGIC
    scatter-summed by doc_ids, with document packing across chunks.

    Exercises the non-trivial path used by the empirical per-token/per-doc
    comparison: chunks contain multiple documents, one document spans two
    chunks, and the per-doc weight is shared across all positions of that
    doc. Mirrors the scatter_add(doc_ids) aggregation in
    scripts/correlate_pertoken_vs_docrun.py.
    """
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            "labels": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            "attention_mask": [[1] * 6, [1] * 6],
            # Packed: 4 unique docs across 2 chunks; doc 2 spans both chunks.
            "doc_ids": [[0, 0, 1, 1, 1, 2], [2, 2, 2, 3, 3, 3]],
        }
    )
    N, T, num_docs = len(ds), 6, 4

    per_doc = _run_magic(model_name, ds, weight_shape=(num_docs,))
    per_tok = _run_magic(model_name, ds, weight_shape=(N, T))

    assert per_doc.shape == (num_docs,)
    assert per_tok.shape == (N, T)

    flat_doc_ids = torch.tensor(ds["doc_ids"]).reshape(-1)
    agg = torch.zeros(num_docs, dtype=torch.float64)
    agg.scatter_add_(0, flat_doc_ids, per_tok.reshape(-1).to(torch.float64))

    # Every doc should receive at least one nonzero token contribution.
    assert (agg.abs() > 0).all(), f"Some doc has zero aggregated score: {agg}"
    torch.testing.assert_close(
        agg, per_doc.to(torch.float64), atol=1e-5, rtol=1e-4
    )


def test_magic_resume(dataset):
    """Resume from a checkpoint mid-training and verify identical final state."""
    device = "cpu"

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
    trainer, fwd_state = Trainer.initialize(model, optimizer)

    # batch_size=1 gives us 2 batches so resume has something to skip
    train_stream = DataStream(dataset, batch_size=1, device=device)
    assert len(train_stream) == 2

    with tempfile.TemporaryDirectory() as ckpt_dir:
        # Full training run (inplace=False to keep fwd_state intact)
        final_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=False,
            save_dir=ckpt_dir,
            save_mode="all",
        )

        # Resume from checkpoints with the same initial state
        resumed_state = trainer.train(
            fwd_state,
            train_stream,
            inplace=False,
            save_dir=ckpt_dir,
            save_mode="all",
            resume=True,
        )

        for k in final_state.params:
            torch.testing.assert_close(resumed_state.params[k], final_state.params[k])
