from __future__ import annotations

import argparse
import sys
from pathlib import Path


_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = next(p for p in (_HERE.parent, *_HERE.parents) if p.name == "scripts")
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _repo import ensure_repo_on_path

_REPO_ROOT = ensure_repo_on_path()


from src.core.visualizer import AggregatedSignalVisualizer, ensure_output_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregated signal visualization")
    default_config = _REPO_ROOT / "config.yaml"
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to YAML config.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on a single sample (first subject-velocity-trial group).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=None,
        help="Aggregation modes to run (default: all enabled).",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=None,
        help="Signal groups to run (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualizer = AggregatedSignalVisualizer(Path(args.config))
    ensure_output_dirs(visualizer.base_dir, visualizer.config)
    visualizer.run(modes=args.modes, signal_groups=args.groups, sample=args.sample)


if __name__ == "__main__":
    main()
