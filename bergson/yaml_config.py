"""Load bergson CLI arguments from a YAML config file.

Usage::

    bergson trackstar --config path/to/config.yaml

The YAML file should contain a flat mapping of CLI argument names (without
leading ``--``) to their values.  Boolean flags can be set to ``true``/
``false``.  The special key ``run_path`` is treated as the positional
argument.

Example YAML::

    run_path: runs/my_experiment
    model: EleutherAI/pythia-160m
    projection_dim: 64
    fsdp: true
    index_cfg.precision: fp32
    data.dataset: wikitext
    data.subset: wikitext-103-raw-v1
    data.split: "train[:10000]"
    data.truncation: true
    query.dataset: cais/wmdp
    query.split: test
    query.subset: wmdp-bio
    query.format_template: bergson/templates/mcqa.yaml
    query.truncation: true
    unit_normalize: true
    aggregation: mean
    overwrite: true

Use dotted keys (``data.dataset: wikitext``) to match CLI flag names
exactly.  Nested YAML mappings are also supported and will be flattened
automatically.  For ambiguous fields like ``precision`` that exist in
multiple configs, use the full prefixed form (``index_cfg.precision``).

CLI arguments given after ``--config`` take precedence over YAML values.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Keys that map to positional arguments rather than --flags.
_POSITIONAL_KEYS = {"run_path"}

# Keys that are boolean flags accepting no value on the CLI.
_BOOL_FLAG_KEYS = {
    "fsdp",
    "truncation",
    "overwrite",
    "unit_normalize",
    "normalize_aggregated_grad",
    "include_bias",
    "reshape_to_square",
    "auto_batch_size",
    "skip_preconditioners",
    "skip_index",
    "drop_columns",
    "profile",
    "debug",
    "attribute_tokens",
    "num_stats_sample_preconditioner",
    "skip_nan_rewards",
    "healthcheck",
}


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a nested dict into ``(dotted_key, str_value)`` pairs."""
    items: list[tuple[str, str]] = []
    for key, value in d.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            items.extend(_flatten(value, full_key))
        else:
            items.append((full_key, value))
    return items


def yaml_to_args(yaml_path: str | Path) -> list[str]:
    """Convert a YAML config file to a list of CLI arguments."""
    path = Path(yaml_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        return []

    args: list[str] = []
    for key, value in _flatten(cfg):
        # Positional args are bare values without --prefix
        if key in _POSITIONAL_KEYS:
            # Insert positional args at the front so they appear right after
            # the subcommand name.
            args.insert(0, str(value))
            continue

        flag = f"--{key}"

        # Boolean handling: simple_parsing uses --flag / --noflag
        bare_key = key.rsplit(".", 1)[-1]
        if bare_key in _BOOL_FLAG_KEYS:
            if value is True or str(value).lower() == "true":
                args.append(flag)
            else:
                args.append(f"--no{key}")
            continue

        args.extend([flag, str(value)])

    return args


def expand_yaml_config(args: list[str]) -> list[str]:
    """If ``--config <path>`` is present in *args*, load the YAML and merge.

    Arguments that appear explicitly on the CLI take precedence over those
    loaded from the YAML file.
    """
    if "--config" not in args:
        return args

    idx = args.index("--config")
    if idx + 1 >= len(args):
        raise ValueError("--config requires a path argument")

    yaml_path = args[idx + 1]
    # Remove --config and its argument from the arg list
    remaining = args[:idx] + args[idx + 2 :]

    yaml_args = yaml_to_args(yaml_path)

    # CLI args come after YAML args so they override via simple_parsing
    # But positional args (run_path) from YAML need to go after the
    # subcommand. We insert yaml_args before the remaining CLI args.
    # The subcommand name (e.g. "trackstar") is the first arg in remaining.
    if remaining:
        # Find where the subcommand ends - it's the first arg
        return [remaining[0]] + yaml_args + remaining[1:]
    return yaml_args
