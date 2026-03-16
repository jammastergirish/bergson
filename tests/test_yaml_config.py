from pathlib import Path

import pytest
import yaml

from bergson.cli_commands import Build, Trackstar
from bergson.yaml_config import load_main_from_yaml


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_build_yaml_infers_command_and_decodes_nested_types(tmp_path: Path):
    config_path = write_yaml(
        tmp_path / "build.yaml",
        {
            "version": 1,
            "command": "build",
            "index": {
                "run_path": "runs/from-yaml",
                "model": "yaml-model",
                "modules": ["layer1", "layer2"],
                "data": {
                    "dataset": "yaml-dataset",
                    "truncation": False,
                },
            },
            "preprocess": {
                "unit_normalize": True,
            },
        },
    )

    prog = load_main_from_yaml(["--config", str(config_path)])

    assert prog is not None
    assert isinstance(prog.command, Build)
    assert prog.command.index_cfg.run_path == "runs/from-yaml"
    assert prog.command.index_cfg.model == "yaml-model"
    assert prog.command.index_cfg.modules == ["layer1", "layer2"]
    assert prog.command.index_cfg.data.dataset == "yaml-dataset"
    assert prog.command.index_cfg.data.truncation is False
    assert prog.command.preprocess_cfg.unit_normalize is True


def test_yaml_inferred_command_supports_negative_bool_cli_override(tmp_path: Path):
    config_path = write_yaml(
        tmp_path / "build.yaml",
        {
            "command": "build",
            "index": {
                "run_path": "runs/from-yaml",
                "data": {
                    "truncation": True,
                },
            },
        },
    )

    prog = load_main_from_yaml(["--config", str(config_path), "--notruncation"])

    assert prog is not None
    assert isinstance(prog.command, Build)
    assert prog.command.index_cfg.run_path == "runs/from-yaml"
    assert prog.command.index_cfg.data.truncation is False


def test_trackstar_cli_overrides_yaml_values(tmp_path: Path):
    config_path = write_yaml(
        tmp_path / "trackstar.yaml",
        {
            "command": "trackstar",
            "index": {
                "run_path": "runs/from-yaml",
                "model": "yaml-model",
                "modules": ["yaml-module"],
                "data": {
                    "dataset": "yaml-index",
                },
            },
            "score": {
                "query_path": "runs/query",
            },
            "preprocess": {
                "unit_normalize": True,
            },
            "trackstar": {
                "query": {
                    "dataset": "yaml-query",
                    "truncation": True,
                },
            },
        },
    )

    prog = load_main_from_yaml(
        [
            "trackstar",
            "--config",
            str(config_path),
            "runs/from-cli",
            "--model",
            "cli-model",
            "--index_cfg.modules",
            "cli-a",
            "cli-b",
            "--query.notruncation",
        ]
    )

    assert prog is not None
    assert isinstance(prog.command, Trackstar)
    assert prog.command.index_cfg.run_path == "runs/from-cli"
    assert prog.command.index_cfg.model == "cli-model"
    assert prog.command.index_cfg.modules == ["cli-a", "cli-b"]
    assert prog.command.score_cfg.query_path == "runs/query"
    assert prog.command.preprocess_cfg.unit_normalize is True
    assert prog.command.trackstar_cfg.query.dataset == "yaml-query"
    assert prog.command.trackstar_cfg.query.truncation is False


def test_yaml_command_mismatch_raises(tmp_path: Path):
    config_path = write_yaml(
        tmp_path / "build.yaml",
        {
            "command": "build",
            "index": {"run_path": "runs/from-yaml"},
        },
    )

    with pytest.raises(ValueError, match="does not match YAML command"):
        load_main_from_yaml(["trackstar", "--config", str(config_path)])
