from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_true

from aif_mojo.numerics import LOG_ZERO, safe_log
from aif_mojo.precise_info_seeking import (
    compute_precise_obs_kernels,
    precise_info_seeking_planning_dense,
    precise_info_seeking_planning_dense_theta_goal,
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


def log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
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


def terminal_plan(iterations: Int) -> List[Float32]:
    return precise_info_seeking_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        iterations,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def test_observation_kernel_uses_information_ratio() raises:
    var observation_values = List[Float32]()
    observation_values.extend(pair(0.9, 0.4))
    observation_values.extend(pair(0.1, 0.6))
    var conditional_values = List[Float32]()
    conditional_values.extend(pair(0.8, 0.3))
    conditional_values.extend(pair(0.2, 0.7))
    var log_observation = log_values(observation_values)
    var log_conditional = log_values(conditional_values)
    var log_marginal = log_values(pair(0.6, 0.4))
    var kernel = compute_precise_obs_kernels(
        log_observation,
        log_conditional,
        log_marginal,
        0,
        1,
        2,
        1,
        2,
    )
    assert_close(kernel[0], safe_log(0.96))
    assert_close(kernel[1], safe_log(0.06))
    assert_close(kernel[2], safe_log(0.01))
    assert_close(kernel[3], safe_log(0.735))


def test_terminal_goal_matches_jax_and_channels_are_conditional() raises:
    var result = terminal_plan(2)
    assert_true(len(result) == 34)
    assert_close(result[0], 0.6513924)
    assert_close(result[1], 0.34860763)
    var expected_action_channels = List[Float32]()
    expected_action_channels.append(-0.5078108)
    expected_action_channels.append(-0.9208302)
    expected_action_channels.append(-0.5114565)
    expected_action_channels.append(-0.9153452)
    expected_action_channels.append(-0.66949964)
    expected_action_channels.append(-0.71736753)
    expected_action_channels.append(-0.4861926)
    expected_action_channels.append(-0.95441675)
    for index in range(8):
        assert_close(result[2 + index], expected_action_channels[index])
    for time_idx in range(2):
        for state_idx in range(2):
            var offset = 2 + (time_idx * 2 + state_idx) * 2
            assert_close(exp(result[offset]) + exp(result[offset + 1]), 1.0)

    assert_close(result[10], -0.06756467)
    assert_close(result[33], -1.8744252)
    for time_idx in range(3):
        for state_idx in range(2):
            for static_idx in range(2):
                var obs_zero = 10 + time_idx * 8 + state_idx * 2 + static_idx
                var obs_one = obs_zero + 4
                assert_close(exp(result[obs_zero]) + exp(result[obs_one]), 1.0)


def test_theta_goal_matches_jax_and_iterations_refine_policy() raises:
    var goal = List[Float32]()
    goal.extend(pair(0.15, 0.75))
    goal.extend(pair(0.85, 0.25))
    var theta_result = precise_info_seeking_planning_dense_theta_goal(
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
    assert_close(theta_result[0], 0.64002913)
    assert_close(theta_result[1], 0.3599709)
    assert_close(theta_result[2], -0.53124076)
    assert_close(theta_result[9], -0.92167515)

    var first = terminal_plan(1)
    assert_close(first[0], 0.59700006)
    assert_close(first[1], 0.40299994)
    assert_true(abs(first[0] - theta_result[0]) > 1.0e-3)


def test_action_mask_is_preserved() raises:
    var result = precise_info_seeking_planning_dense(
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
    for index in range(4):
        assert_close(result[3 + index * 2], LOG_ZERO, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
