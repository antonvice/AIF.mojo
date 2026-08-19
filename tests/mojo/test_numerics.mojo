from std.testing import TestSuite, assert_equal, assert_true
from std.collections import List

from aif_mojo.numerics import (
    LOG_ZERO,
    logaddexp,
    logsumexp,
    normalize_probability,
    safe_log,
    safe_log_div,
    softmax,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def test_safe_log_maps_zero_and_negative_to_log_zero() raises:
    assert_equal(safe_log(0.0), LOG_ZERO)
    assert_equal(safe_log(-1.0), LOG_ZERO)


def test_safe_log_matches_natural_log_for_positive_values() raises:
    assert_close(safe_log(1.0), 0.0)
    assert_close(safe_log(0.5), -0.6931472)


def test_safe_log_clamps_positive_underflow() raises:
    assert_close(safe_log(1.0e-35), -69.07755, 1.0e-4)


def test_safe_log_div_subtracts_only_valid_logs() raises:
    assert_close(safe_log_div(-2.0, -3.0), 1.0)
    assert_equal(safe_log_div(LOG_ZERO, -3.0), LOG_ZERO)
    assert_equal(safe_log_div(-2.0, LOG_ZERO), LOG_ZERO)
    assert_equal(safe_log_div(LOG_ZERO, LOG_ZERO), LOG_ZERO)


def test_logsumexp_is_stable_for_large_logits() raises:
    var values = List[Float32]()
    values.append(1000.0)
    values.append(1000.0)
    assert_close(logsumexp(values), 1000.6931472, 1.0e-4)


def test_logaddexp_handles_log_zero_and_large_values() raises:
    assert_close(logaddexp(LOG_ZERO, safe_log(0.25)), safe_log(0.25))
    assert_close(logaddexp(1000.0, 1000.0), 1000.6931472, 1.0e-4)


def test_softmax_normalizes_logits() raises:
    var logits = List[Float32]()
    logits.append(1.0)
    logits.append(2.0)
    logits.append(3.0)
    var result = softmax(logits)
    assert_close(result[0], 0.09003057)
    assert_close(result[1], 0.24472848)
    assert_close(result[2], 0.66524094)
    assert_close(result[0] + result[1] + result[2], 1.0)


def test_probability_normalization_matches_jax_epsilon_contract() raises:
    var values = List[Float32]()
    values.append(2.0)
    values.append(3.0)
    var result = normalize_probability(values)
    assert_close(result[0], 0.4)
    assert_close(result[1], 0.6)

    var zeros = List[Float32]()
    zeros.append(0.0)
    zeros.append(0.0)
    var normalized_zeros = normalize_probability(zeros)
    assert_equal(normalized_zeros[0], 0.0)
    assert_equal(normalized_zeros[1], 0.0)


def test_all_impossible_logits_reduce_finitely_and_softmax_uniformly() raises:
    var values = List[Float32]()
    values.append(LOG_ZERO)
    values.append(LOG_ZERO)
    assert_equal(logsumexp(values), LOG_ZERO)
    var result = softmax(values)
    assert_close(result[0], 0.5)
    assert_close(result[1], 0.5)


def test_empty_reductions_have_explicit_contract() raises:
    var values = List[Float32]()
    assert_equal(logsumexp(values), LOG_ZERO)
    assert_equal(len(softmax(values)), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
