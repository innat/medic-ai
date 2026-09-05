"""Benchmark MedicAI transforms on CPU and available accelerators."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from .common.devices import devices
    from .common.reporting import format_result
    from .transform_benchmark.runner import profile
    from .transform_benchmark.specs import transform_specs
else:
    from common.devices import devices
    from common.reporting import format_result
    from transform_benchmark.runner import profile
    from transform_benchmark.specs import transform_specs


def main() -> None:
    """Parse CLI options and run the selected transform benchmark suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument("--group", choices=("cpu", "cpu+gpu", "all"), default="all")
    parser.add_argument(
        "--transform",
        nargs="+",
        default=["all"],
        metavar="NAME",
        help=(
            "Transform name(s) to benchmark. Use 'all' (default) to run the "
            "complete suite."
        ),
    )
    parser.add_argument("--layout", choices=("HWC", "DHWC", "BHWC", "BDHWC"), default="BDHWC")
    parser.add_argument("--sizes", type=int, nargs="+", help="Square 2D or cubic 3D spatial sizes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", choices=("none", "xla"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.batch_size < 1 or args.channels < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error(
            "--batch-size, --channels, and --iterations must be positive; "
            "--warmup must be non-negative."
        )
    default_size = 224 if args.layout in ("HWC", "BHWC") else 96
    sizes = args.sizes or [default_size]
    if any(size < 1 for size in sizes):
        parser.error("--sizes must contain positive integers.")

    results = []
    for spatial_size in sizes:
        try:
            specs = _select_specs(
                transform_specs(args.layout, spatial_size),
                args.transform,
                layout=args.layout,
            )
        except ValueError as error:
            parser.error(str(error))
        for spec in specs:
            if args.group != "all" and spec.group != args.group:
                continue
            for device in devices(args.device):
                try:
                    result = profile(
                        spec,
                        args.layout,
                        device,
                        spatial_size,
                        args.batch_size,
                        args.channels,
                        args.iterations,
                        args.warmup,
                        args.seed,
                        args.compile,
                    )
                except RuntimeError as error:
                    print(f"SKIP {spec.name:24} {device:10}: {error}")
                    continue
                results.append(result)
                print(format_result(result))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


def _select_specs(specs: Sequence, requested: Sequence[str], *, layout: str) -> list:
    """Filter benchmark specs by case-insensitive public transform name."""
    requested_names = [name for name in requested if name.lower() != "all"]
    if not requested_names:
        return list(specs)

    specs_by_name = {spec.name.lower(): spec for spec in specs}
    selected = []
    missing = []
    for name in requested_names:
        spec = specs_by_name.get(name.lower())
        if spec is None:
            missing.append(name)
        elif spec not in selected:
            selected.append(spec)
    if missing:
        available = ", ".join(spec.name for spec in specs)
        raise ValueError(
            f"Transform(s) {', '.join(missing)!r} are not available for layout "
            f"{layout!r}. Available transforms: {available}."
        )
    return selected


if __name__ == "__main__":
    main()
