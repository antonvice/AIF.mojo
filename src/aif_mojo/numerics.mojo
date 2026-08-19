from std.collections import List
from std.math import exp, log


comptime LOG_ZERO = Float32(-1.0e12)
comptime LOG_ZERO_THRESHOLD = LOG_ZERO / 2.0
comptime MIN_POSITIVE = Float32(1.0e-30)
comptime EPSILON = Float32(1.0e-10)


def safe_log(x: Float32) -> Float32:
    """Natural log with the JAX implementation's finite zero sentinel."""
    if x <= 0.0:
        return LOG_ZERO
    return log(max(x, MIN_POSITIVE))


def safe_log_div(log_num: Float32, log_den: Float32) -> Float32:
    """Subtract valid log values; map zero-valued operands to LOG_ZERO."""
    if log_num > LOG_ZERO_THRESHOLD and log_den > LOG_ZERO_THRESHOLD:
        return log_num - log_den
    return LOG_ZERO


def logaddexp(lhs: Float32, rhs: Float32) -> Float32:
    """Stable log(exp(lhs) + exp(rhs)), preserving the finite zero sentinel."""
    if lhs <= LOG_ZERO_THRESHOLD:
        return rhs
    if rhs <= LOG_ZERO_THRESHOLD:
        return lhs
    var maximum = max(lhs, rhs)
    return maximum + log(exp(lhs - maximum) + exp(rhs - maximum))


def logsumexp(values: List[Float32]) -> Float32:
    """Stable log(sum(exp(values))) for a one-dimensional message."""
    if len(values) == 0:
        return LOG_ZERO

    var maximum = values[0]
    for i in range(1, len(values)):
        maximum = max(maximum, values[i])

    var shifted_sum = Float32(0.0)
    for value in values:
        shifted_sum += exp(value - maximum)
    return maximum + log(shifted_sum)


def softmax(values: List[Float32]) -> List[Float32]:
    """Stable softmax over a one-dimensional log message."""
    var result = List[Float32]()
    if len(values) == 0:
        return result^

    var maximum = values[0]
    for i in range(1, len(values)):
        maximum = max(maximum, values[i])

    var total = Float32(0.0)
    for value in values:
        var shifted = exp(value - maximum)
        result.append(shifted)
        total += shifted
    for i in range(len(result)):
        result[i] /= total
    return result^


def normalize_probability(values: List[Float32]) -> List[Float32]:
    """Normalize as JAX messages.py does: value / (sum + EPSILON)."""
    var total = Float32(0.0)
    for value in values:
        total += value

    var result = List[Float32]()
    var denominator = total + EPSILON
    for value in values:
        result.append(value / denominator)
    return result^
