from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_true

from aif_mojo.region_extended_loopy_bp import (
    region_extended_loopy_bp_planning_dense,
    region_extended_loopy_bp_planning_dense_theta_goal,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def transition() -> List[Float32]:
    # Shape (new state, old state, theta, action).
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.4, 0.7))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.6, 0.1))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.6, 0.3))
    result.extend(pair(0.7, 0.2))
    result.extend(pair(0.4, 0.9))
    return result^


def observation() -> List[Float32]:
    # Shape (field of view, observation, state, theta).
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def terminal_plan(iterations: Int, damping: Float32 = 0.5) -> List[Float32]:
    return region_extended_loopy_bp_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        iterations,
        damping,
        2,
        2,
        2,
        1,
        2,
    )


def test_terminal_goal_matches_jax_and_channel_layout() raises:
    var result = terminal_plan(2)
    # action (2), dynamic channels (2*2*2*2), observation channels (3*1*2*2*2)
    assert_true(len(result) == 42)
    assert_close(result[0], 0.59889793)
    assert_close(result[1], 0.40110204)

    var expected_dynamic = List[Float32]()
    expected_dynamic.append(-0.48989448)
    expected_dynamic.append(-0.72253996)
    expected_dynamic.append(-0.94853270)
    expected_dynamic.append(-0.66459376)
    expected_dynamic.append(-0.71209610)
    expected_dynamic.append(-0.66205347)
    expected_dynamic.append(-0.67455065)
    expected_dynamic.append(-0.72523880)
    expected_dynamic.append(-0.89025283)
    expected_dynamic.append(-1.22575247)
    expected_dynamic.append(-0.52856869)
    expected_dynamic.append(-0.34748411)
    expected_dynamic.append(-1.21138346)
    expected_dynamic.append(-1.14172387)
    expected_dynamic.append(-0.35351568)
    expected_dynamic.append(-0.38458684)
    for index in range(16):
        assert_close(result[2 + index], expected_dynamic[index])

    # Dynamic channels are r(new | old, action).
    for time_idx in range(2):
        for old_idx in range(2):
            for action_idx in range(2):
                var zero_offset = (
                    2 + ((time_idx * 2 + old_idx) * 2) * 2 + action_idx
                )
                var one_offset = zero_offset + 2
                assert_close(
                    exp(result[zero_offset]) + exp(result[one_offset]), 1.0
                )

    assert_close(result[18], -0.06756467)
    assert_close(result[41], -1.87442517)
    # Observation channels are r(observation | state, theta).
    for time_idx in range(3):
        for state_idx in range(2):
            for static_idx in range(2):
                var obs_zero = 18 + time_idx * 8 + state_idx * 2 + static_idx
                var obs_one = obs_zero + 4
                assert_close(exp(result[obs_zero]) + exp(result[obs_one]), 1.0)


def test_theta_goal_matches_jax() raises:
    var goal = List[Float32]()
    goal.extend(pair(0.15, 0.75))
    goal.extend(pair(0.85, 0.25))
    var result = region_extended_loopy_bp_planning_dense_theta_goal(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        goal,
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 0.59449089)
    assert_close(result[1], 0.40550914)
    assert_close(result[2], -0.54001343)
    assert_close(result[17], -0.66786349)
    assert_close(result[18], -0.06756467)
    assert_close(result[41], -1.87442517)


def test_damped_iterations_refine_policy() raises:
    var first = terminal_plan(1)
    var second = terminal_plan(2)
    assert_close(first[0], 0.59700006)
    assert_close(first[1], 0.40299994)
    assert_close(first[2], -0.54486841)
    assert_close(first[17], -0.34193221)
    assert_true(abs(first[0] - second[0]) > 1.0e-3)


def test_action_mask_is_preserved() raises:
    var result = region_extended_loopy_bp_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_true(result[1] < 1.0e-6)
    # A masked action still has a valid conditional dynamics channel.
    for time_idx in range(2):
        for old_idx in range(2):
            var zero_offset = 2 + ((time_idx * 2 + old_idx) * 2) * 2 + 1
            assert_close(result[zero_offset], -0.69314718)
            assert_close(result[zero_offset + 2], -0.69314718)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
