"""Benchmark MedicAI transforms on CPU and available accelerators."""

import argparse
import json
from pathlib import Path

try:
    from .transform_benchmark.devices import devices
    from .transform_benchmark.runner import profile
    from .transform_benchmark.specs import transform_specs
except ImportError:
    from transform_benchmark.devices import devices
    from transform_benchmark.runner import profile
    from transform_benchmark.specs import transform_specs


def main() -> None:
    """Parse CLI options and run the selected transform benchmark suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument("--group", choices=("cpu", "cpu+gpu", "all"), default="all")
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

    if args.batch_size < 1 or args.channels < 1:
        parser.error("--batch-size and --channels must be positive.")
    default_size = 224 if args.layout in ("HWC", "BHWC") else 96
    sizes = args.sizes or [default_size]
    if any(size < 1 for size in sizes):
        parser.error("--sizes must contain positive integers.")

    results = []
    for spatial_size in sizes:
        for spec in transform_specs(args.layout, spatial_size):
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
                if result["forward_median_ms"] is None:
                    print(
                        f"{spec.name:24} {device:10} {args.layout:6} size={spatial_size:<4} "
                        f"forward={result['compile_status']} inverse={result['inverse_status']} "
                        f"compile={result['compile_status']}"
                    )
                    continue
                inverse = result["inverse_median_ms"]
                inverse_display = (
                    result["inverse_status"] if inverse is None else f"{inverse:.2f} ms"
                )
                compile_display = (
                    "-"
                    if result["compile_time_ms"] is None
                    else f"{result['compile_time_ms']:.2f} ms"
                )
                print(
                    f"{spec.name:24} {device:10} {args.layout:6} size={spatial_size:<4} "
                    f"forward={result['forward_median_ms']:.2f} ms "
                    f"inverse={inverse_display} compile={compile_display}"
                )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
