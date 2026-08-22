from std.collections import List


def max_channel_residual(
    old_channels: List[Float32], new_channels: List[Float32]
) -> Float32:
    """Maximum absolute log-channel change for one fixed-point update."""
    debug_assert(
        len(old_channels) == len(new_channels), "channel shape mismatch"
    )
    var residual = Float32(0.0)
    for index in range(len(old_channels)):
        residual = max(residual, abs(new_channels[index] - old_channels[index]))
    return residual


def combined_channel_residual(
    old_first: List[Float32],
    new_first: List[Float32],
    old_second: List[Float32],
    new_second: List[Float32],
) -> Float32:
    """Maximum residual across two independently normalized channels."""
    return max(
        max_channel_residual(old_first, new_first),
        max_channel_residual(old_second, new_second),
    )


def next_adaptive_damping(
    current: Float32,
    previous_residual: Float32,
    residual: Float32,
    minimum: Float32 = 0.05,
    maximum: Float32 = 1.0,
) -> Float32:
    """Conservative residual controller for the next fixed-point update.

    A residual increase halves damping. Strong progress raises it by ten
    percent. The first iteration and small changes retain the current value.
    """
    debug_assert(minimum > 0.0, "minimum damping must be positive")
    debug_assert(maximum >= minimum, "invalid damping bounds")
    var updated = min(max(current, minimum), maximum)
    if previous_residual < 0.0:
        return updated
    if residual > previous_residual * 1.05:
        return max(minimum, updated * 0.5)
    if residual < previous_residual * 0.5:
        return min(maximum, updated * 1.1)
    return updated


def append_convergence_metadata(
    mut result: List[Float32],
    converged: Bool,
    iterations_used: Int,
    final_residual: Float32,
    final_damping: Float32,
    residual_history: List[Float32],
):
    """Append `[converged, iterations, residual, damping, history...]`."""
    result.append(Float32(1.0 if converged else 0.0))
    result.append(Float32(iterations_used))
    result.append(final_residual)
    result.append(final_damping)
    for value in residual_history:
        result.append(value)
