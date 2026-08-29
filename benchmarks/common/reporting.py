"""Console formatting helpers for benchmark results."""


def format_result(result: dict) -> str:
    """Format one benchmark result for the console."""
    if result["forward_median_ms"] is None:
        return (
            f"{result['transform']:24} {result['device']:10} {result['layout']:6} "
            f"size={result['spatial_size']:<4} "
            f"forward={result['compile_status']} "
            f"inverse={result['inverse_status']} "
            f"compile={result['compile_status']}"
        )

    inverse = result["inverse_median_ms"]
    inverse_display = (
        result["inverse_status"] if inverse is None else f"{inverse:.2f} ms"
    )
    compile_time = result["compile_time_ms"]
    compile_display = "-" if compile_time is None else f"{compile_time:.2f} ms"
    return (
        f"{result['transform']:24} {result['device']:10} {result['layout']:6} "
        f"size={result['spatial_size']:<4} "
        f"forward={result['forward_median_ms']:.2f} ms "
        f"inverse={inverse_display} compile={compile_display}"
    )
