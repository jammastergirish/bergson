"""Load structured Bergson CLI configs from YAML.

Usage::

    bergson --config path/to/config.yaml
    bergson trackstar --config path/to/config.yaml

The YAML file is structured around Bergson's config groups rather than raw CLI
flags. A config declares its command explicitly and nests values under sections
such as ``index``, ``preprocess``, and ``score``.

Example YAML::

    version: 1
    command: trackstar
    index:
      run_path: runs/my_experiment
      model: EleutherAI/pythia-160m
      data:
        dataset: wikitext
        split: "train[:10000]"
        truncation: true
    preprocess:
      unit_normalize: true
      aggregation: mean
    score:
      query_path: runs/query
    trackstar:
      query:
        dataset: cais/wmdp
        split: test
        subset: wmdp-bio
        truncation: true

The CLI command may be omitted when ``command`` is present in the YAML. If both
the CLI and YAML specify a command, they must match. Explicit CLI arguments
override YAML values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from simple_parsing import ArgumentParser
from simple_parsing.helpers.serialization.serializable import from_dict

from .cli_commands import (
    Build,
    Hessian,
    Main,
    Preconditioners,
    Query,
    Reduce,
    Score,
    Trackstar,
    build_main_parser,
)

_CONFIG_FLAG = "--config"
_HELP_FLAGS = {"-h", "--help"}
@dataclass(frozen=True)
class CommandSpec:
    command_cls: type
    section_aliases: dict[str, str]
    positional_path: tuple[str, ...] | None = None


_COMMAND_SPECS: dict[str, CommandSpec] = {
    "build": CommandSpec(
        command_cls=Build,
        section_aliases={"index": "index_cfg", "preprocess": "preprocess_cfg"},
        positional_path=("index_cfg", "run_path"),
    ),
    "preconditioners": CommandSpec(
        command_cls=Preconditioners,
        section_aliases={"index": "index_cfg"},
        positional_path=("index_cfg", "run_path"),
    ),
    "reduce": CommandSpec(
        command_cls=Reduce,
        section_aliases={"index": "index_cfg", "preprocess": "preprocess_cfg"},
        positional_path=("index_cfg", "run_path"),
    ),
    "score": CommandSpec(
        command_cls=Score,
        section_aliases={
            "index": "index_cfg",
            "score": "score_cfg",
            "preprocess": "preprocess_cfg",
        },
        positional_path=("index_cfg", "run_path"),
    ),
    "query": CommandSpec(
        command_cls=Query,
        section_aliases={"query": "query_cfg"},
    ),
    "hessian": CommandSpec(
        command_cls=Hessian,
        section_aliases={"index": "index_cfg", "hessian": "hessian_cfg"},
        positional_path=("index_cfg", "run_path"),
    ),
    "trackstar": CommandSpec(
        command_cls=Trackstar,
        section_aliases={
            "index": "index_cfg",
            "score": "score_cfg",
            "preprocess": "preprocess_cfg",
            "trackstar": "trackstar_cfg",
        },
        positional_path=("index_cfg", "run_path"),
    ),
}


def _split_config_arg(args: list[str]) -> tuple[Path | None, list[str]]:
    if _CONFIG_FLAG not in args:
        return None, args

    idx = args.index(_CONFIG_FLAG)
    if idx + 1 >= len(args):
        raise ValueError("--config requires a path argument")

    return Path(args[idx + 1]), args[:idx] + args[idx + 2 :]


def _extract_cli_command(args: list[str]) -> str | None:
    if args and args[0] in _COMMAND_SPECS:
        return args[0]
    return None


def _load_yaml_document(path: Path) -> dict[str, Any]:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}

    if not isinstance(doc, dict):
        raise ValueError(f"YAML config at {path} must contain a top-level mapping")

    version = doc.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported YAML config version {version!r} in {path}")

    return doc


def _normalize_yaml_sections(doc: dict[str, Any], command: str) -> dict[str, Any]:
    spec = _COMMAND_SPECS[command]
    allowed_keys = {"version", "command", *spec.section_aliases.keys()}
    unknown_keys = sorted(set(doc) - allowed_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(
            f"Unknown top-level YAML keys for command '{command}': {joined}"
        )

    payload: dict[str, Any] = {}
    for yaml_key, field_name in spec.section_aliases.items():
        if yaml_key not in doc:
            continue

        value = doc[yaml_key]
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(
                f"Top-level YAML key '{yaml_key}' must contain a mapping, got "
                f"{type(value).__name__}"
            )
        payload[field_name] = value

    return payload


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_nested(obj: dict[str, Any], path: tuple[str, ...] | None) -> Any:
    if path is None:
        return None

    current: Any = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(obj: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = obj
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _command_skeleton(command_name: str) -> dict[str, Any]:
    command_cls = _COMMAND_SPECS[command_name].command_cls
    return {field.name: {} for field in fields(command_cls)}


def _show_help(command_name: str | None) -> None:
    help_args = [command_name, "--help"] if command_name else ["--help"]
    build_main_parser().parse_args(help_args)


def _matches_option(arg: str, option: str) -> bool:
    return arg == option or arg.startswith(f"{option}=")


def _action_used_on_cli(action: Any, cli_args: list[str]) -> bool:
    return any(
        _matches_option(arg, option)
        for arg in cli_args
        for option in getattr(action, "option_strings", [])
    )


def _dest_to_path(dest: str) -> tuple[str, ...]:
    parts = dest.split(".")
    while parts and parts[0] in {"cfg", "prog", "command"}:
        parts = parts[1:]
    return tuple(parts)


def _get_command_subparser(parser: ArgumentParser, command_name: str) -> Any:
    parser._preprocessing([command_name])
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices and command_name in choices:
            return choices[command_name]
    raise ValueError(f"Unable to find subparser for command '{command_name}'")


def _parse_cli_patch(
    command_name: str,
    cli_args: list[str],
    yaml_payload: dict[str, Any],
) -> dict[str, Any]:
    spec = _COMMAND_SPECS[command_name]
    parser = build_main_parser()
    subparser = _get_command_subparser(parser, command_name)
    yaml_positional = _get_nested(yaml_payload, spec.positional_path)

    try:
        parsed_cfg = parser.parse_args(args=[command_name, *cli_args]).prog.command
    except SystemExit:
        if yaml_positional is None:
            raise
        parsed_cfg = parser.parse_args(
            args=[command_name, str(yaml_positional), *cli_args]
        ).prog.command

    parsed_dict = asdict(parsed_cfg)
    patch_dict: dict[str, Any] = {}

    for action in subparser._actions:
        if action.dest == "help" or not getattr(action, "option_strings", []):
            continue
        if not _action_used_on_cli(action, cli_args):
            continue

        path = _dest_to_path(action.dest)
        if not path:
            continue
        value = _get_nested(parsed_dict, path)
        _set_nested(patch_dict, path, value)

    if spec.positional_path is not None:
        parsed_positional = _get_nested(parsed_dict, spec.positional_path)
        if parsed_positional is not None and (
            yaml_positional is None or parsed_positional != yaml_positional
        ):
            _set_nested(patch_dict, spec.positional_path, parsed_positional)

    return patch_dict


def load_main_from_yaml(args: list[str]) -> Main | None:
    """Load and merge a structured YAML config if ``--config`` is present."""
    config_path, remaining_args = _split_config_arg(list(args))
    if config_path is None:
        return None

    cli_command = _extract_cli_command(remaining_args)
    cli_args = remaining_args[1:] if cli_command is not None else remaining_args

    doc = _load_yaml_document(config_path)
    yaml_command = doc.get("command")
    if yaml_command is not None and yaml_command not in _COMMAND_SPECS:
        raise ValueError(f"Unknown Bergson command in YAML config: {yaml_command!r}")

    resolved_command = cli_command or yaml_command
    if resolved_command is None:
        raise ValueError(
            "YAML config must declare a 'command' when the CLI command is omitted"
        )

    if (
        cli_command is not None
        and yaml_command is not None
        and cli_command != yaml_command
    ):
        raise ValueError(
            f"CLI command '{cli_command}' does not match YAML command '{yaml_command}'"
        )

    if any(arg in _HELP_FLAGS for arg in remaining_args):
        _show_help(resolved_command)

    yaml_payload = _normalize_yaml_sections(doc, resolved_command)
    cli_patch = _parse_cli_patch(resolved_command, cli_args, yaml_payload)

    merged_payload = _deep_merge(_command_skeleton(resolved_command), yaml_payload)
    merged_payload = _deep_merge(merged_payload, cli_patch)

    command_cfg = from_dict(
        _COMMAND_SPECS[resolved_command].command_cls, merged_payload
    )
    return Main(command=command_cfg)
