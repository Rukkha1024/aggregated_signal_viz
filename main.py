from pathlib import Path

from script.visualizer import AggregatedSignalVisualizer, ensure_output_dirs, parse_args


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    visualizer = AggregatedSignalVisualizer(config_path)
    ensure_output_dirs(visualizer.base_dir, visualizer.config)
    visualizer.run(modes=args.modes, signal_groups=args.groups, sample=args.sample)


if __name__ == "__main__":
    main()
