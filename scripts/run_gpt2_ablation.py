import argparse
import os
import pathlib
import runpy
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--num-train-samples", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-test-samples", type=int, default=1)
    parser.add_argument("--noac", action="store_true")
    parser.add_argument("--untie-lm-head", action="store_true")
    parser.add_argument("--disable-trainmode", action="store_true")
    parser.add_argument("--disable-cuda-rng", action="store_true")
    parser.add_argument("--disable-loadfix", action="store_true")
    parser.add_argument("--disable-trace-functional-call", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.noac:
        torch.cuda.is_bf16_supported = lambda *a, **k: False

    import bergson.trainer as bt
    import transformers

    if args.disable_trainmode:
        @contextmanager
        def passthrough_training_mode(model, training):
            yield

        bt._set_training_mode = passthrough_training_mode

    if args.disable_cuda_rng:
        bt._set_cuda_rng_state = lambda state, device: None

    if args.disable_loadfix:
        def old_load(self, path: str):
            dcp.load(
                self.state_dict(),
                checkpoint_id=path,
                no_dist=not dist.is_initialized(),
            )

        bt.TrainerState.load = old_load

    if args.disable_trace_functional_call:
        def old_trace_step(self, state, inputs, *, inplace=False, trace=False):
            device = next(iter(state.params.values())).device
            torch.random.set_rng_state(state.cpu_rng_state)
            bt._set_cuda_rng_state(state.cuda_rng_state, device)
            alias_groups = bt._parameter_alias_groups(state.params)
            canonical_params = bt._canonical_param_dict(state.params, alias_groups)

            with (
                bt._set_training_mode(self.model, True),
                torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_bf16_supported(),
                ),
                bt.swap_parameters(self.model, state.params, state.buffers) as params,
            ):
                outputs = self.model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs
                assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
                grads = bt.grad_tree(loss, params, create_graph=trace)

            canonical_keys, canonical_grads = bt._canonicalize_param_grads(
                grads, alias_groups
            )
            updates, new_state = self.optimizer.update(
                dict(zip(canonical_keys, canonical_grads, strict=True)),
                state.opt_state,
                inplace=inplace,
                params=canonical_params,
            )
            new_params = bt._apply_updates_with_aliases(
                state.params,
                updates,
                alias_groups,
                inplace=inplace,
            )
            new_params = bt._retie_aliases(new_params, alias_groups)
            return bt.TrainerState(
                new_params,
                new_state,
                state.buffers,
                state.batch_index + 1,
                cuda_rng_state=bt._maybe_get_cuda_rng_state(),
                cpu_rng_state=torch.random.get_rng_state(),
            )

        bt.Trainer.step = old_trace_step

    if args.untie_lm_head:
        original_from_pretrained = transformers.GPT2LMHeadModel.from_pretrained

        def untied_from_pretrained(*model_args, **model_kwargs):
            model = original_from_pretrained(*model_args, **model_kwargs)
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
            return model

        transformers.GPT2LMHeadModel.from_pretrained = untied_from_pretrained

    script = pathlib.Path(__file__).resolve().parents[1] / "examples" / "gpt2_wikitext.py"
    sys_argv = [
        str(script),
        "score",
        "--save_dir",
        args.save_dir,
        "--num_train_samples",
        str(args.num_train_samples),
        "--num_epochs",
        str(args.num_epochs),
        "--num_test_samples",
        str(args.num_test_samples),
        "--nproc_per_node",
        "1",
    ]
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    import sys

    sys.argv = sys_argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
