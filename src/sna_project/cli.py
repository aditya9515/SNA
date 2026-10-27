import argparse
from pathlib import Path

from .data import prepare_all
from .experiment import aggregate, run_experiment, select_parameter


def main():
    parser = argparse.ArgumentParser(description="Reproduce community-detection experiments")
    parser.add_argument("command", choices=["prepare", "select", "run", "summarize", "plot"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        prepare_all(root)
    elif args.command == "select":
        select_parameter(root)
    elif args.command == "run":
        if args.config is None:
            parser.error("run requires --config")
        config = args.config if args.config.is_absolute() else root / args.config
        print(run_experiment(root, config))
    elif args.command == "summarize":
        aggregate(root)
    else:
        from .figures import build_figures
        build_figures(root)
