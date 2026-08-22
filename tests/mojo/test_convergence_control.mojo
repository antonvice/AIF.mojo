from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.convergence_control import (
    append_convergence_metadata,
    combined_channel_residual,
    max_channel_residual,
    next_adaptive_damping,
)


def test_channel_residuals_use_maximum_absolute_log_change() raises:
    var old = List[Float32]()
    var new = List[Float32]()
    for value in [Float32(0.0), -1.0, -4.0]:
        old.append(value)
    for value in [Float32(0.25), -1.5, -3.9]:
        new.append(value)
    assert_true(abs(max_channel_residual(old, new) - 0.5) < 1.0e-6)

    var old_second = List[Float32]()
    var new_second = List[Float32]()
    for value in [Float32(0.0), -2.0]:
        old_second.append(value)
    for value in [Float32(0.0), -2.75]:
        new_second.append(value)
    assert_true(
        abs(combined_channel_residual(old, new, old_second, new_second) - 0.75)
        < 1.0e-6
    )


def test_adaptive_damping_reacts_to_residual_progress() raises:
    assert_true(abs(next_adaptive_damping(0.8, 1.0, 1.2) - 0.4) < 1.0e-6)
    assert_true(abs(next_adaptive_damping(0.5, 1.0, 0.4) - 0.55) < 1.0e-6)
    assert_true(abs(next_adaptive_damping(0.5, -1.0, 8.0) - 0.5) < 1.0e-6)
    assert_true(abs(next_adaptive_damping(0.05, 1.0, 2.0) - 0.05) < 1.0e-6)


def test_metadata_layout_is_explicit_and_history_preserving() raises:
    var result = List[Float32]()
    result.append(0.2)
    result.append(0.8)
    var history = List[Float32]()
    history.append(1.0)
    history.append(0.25)
    append_convergence_metadata(result, True, 2, 0.25, 0.4, history)
    assert_equal(len(result), 8)
    assert_equal(result[2], 1.0)
    assert_equal(result[3], 2.0)
    assert_equal(result[4], 0.25)
    assert_equal(result[5], 0.4)
    assert_equal(result[6], 1.0)
    assert_equal(result[7], 0.25)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
