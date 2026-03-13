import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import torch
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--num-test-samples", type=int, default=1)
    parser.add_argument("--num-subsets", type=int, default=10)
    parser.add_argument("--drop-fraction", type=float, default=0.01)
    parser.add_argument("--noac", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def score_stats(scores_path: pathlib.Path) -> dict[str, float | int | bool]:
    scores = torch.load(scores_path, map_location="cpu")
    flat = scores.reshape(-1).float()
    q = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    return {
        "shape": tuple(scores.shape),
        "dtype": str(scores.dtype),
        "finite": bool(torch.isfinite(flat).all()),
        "nonzero": int((flat != 0).sum().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "absmax": float(flat.abs().max().item()),
        "q01": float(q[0].item()),
        "q05": float(q[1].item()),
        "median": float(q[2].item()),
        "q95": float(q[3].item()),
        "q99": float(q[4].item()),
    }


def lds_result(results_path: pathlib.Path) -> dict[str, float | int]:
    data = json.loads(results_path.read_text())
    pred = data["predicted"][0]
    true = data["true"][0]
    return {
        "rho": float(spearmanr(pred, true).statistic),
        "n_subsets": int(data["n_subsets"]),
        "pred_min": float(min(pred)),
        "pred_max": float(max(pred)),
        "true_min": float(min(true)),
        "true_max": float(max(true)),
    }


def main():
    args = parse_args()
    save_dir = pathlib.Path(args.save_dir)
    scores_path = save_dir / "scores.pt"
    config_path = save_dir / "config.json"
    lds_path = save_dir / f"lds_drop{int(args.drop_fraction * 100)}_s0.json"

    while not (scores_path.exists() and config_path.exists()):
        time.sleep(args.poll_seconds)

    stats = score_stats(scores_path)
    print(json.dumps({"save_dir": str(save_dir), "score_stats": stats}, sort_keys=True), flush=True)

    if not lds_path.exists():
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["PYTHONUNBUFFERED"] = "1"

        if args.noac:
            cmd = [
                "python3",
                "-c",
                (
                    "import runpy,sys,torch; "
                    "torch.cuda.is_bf16_supported=lambda *a,**k: False; "
                    "sys.argv=["
                    "\"examples/gpt2_wikitext.py\",\"lds\","
                    f"\"--save_dir\",\"{save_dir}\","
                    f"\"--num_epochs\",\"{args.num_epochs}\","
                    f"\"--num_test_samples\",\"{args.num_test_samples}\","
                    f"\"--num_subsets\",\"{args.num_subsets}\","
                    f"\"--drop_fractions\",\"{args.drop_fraction}\","
                    "\"--nproc_per_node\",\"1\"]; "
                    "runpy.run_path(\"examples/gpt2_wikitext.py\", run_name=\"__main__\")"
                ),
            ]
        else:
            cmd = [
                "python3",
                "examples/gpt2_wikitext.py",
                "lds",
                "--save_dir",
                str(save_dir),
                "--num_epochs",
                str(args.num_epochs),
                "--num_test_samples",
                str(args.num_test_samples),
                "--num_subsets",
                str(args.num_subsets),
                "--drop_fractions",
                str(args.drop_fraction),
                "--nproc_per_node",
                "1",
            ]

        subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    result = lds_result(lds_path)
    print(json.dumps({"save_dir": str(save_dir), "lds": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
