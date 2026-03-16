import sys
from typing import Optional

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
from .yaml_config import load_main_from_yaml

__all__ = [
    "Build",
    "Hessian",
    "Main",
    "Preconditioners",
    "Query",
    "Reduce",
    "Score",
    "Trackstar",
    "main",
]


def main(args: Optional[list[str]] = None):
    """Parse CLI arguments and dispatch to the selected subcommand.

    Supports ``--config path/to/config.yaml`` to load a structured YAML config.
    The YAML can declare its own ``command`` and may be used as either
    ``bergson --config config.yaml`` or ``bergson <command> --config config.yaml``.
    Explicit CLI arguments override YAML values.
    """
    if args is None:
        args = sys.argv[1:]

    configured = load_main_from_yaml(args)
    if configured is not None:
        configured.execute()
        return

    prog: Main = build_main_parser().parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
