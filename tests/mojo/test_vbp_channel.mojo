from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.vbp_channel import (
    vbp_channel_planning_dense,
    vbp_channel_planning_dense_theta_goal,
    vbp_channel_planning_sparse,
    vbp_channel_planning_sparse_theta_goal,
)


def assert_close(actual: Float32, expected: Float32) raises:
    assert_true(abs(actual - expected) <= 1.0e-5)


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def transition_indices() -> List[Int]:
    var result = List[Int]()
    result.append(0)
    result.append(1)
    result.append(1)
    result.append(0)
    result.append(1)
    result.append(0)
    result.append(0)
    result.append(1)
    return result^


def dense_transition() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(1.0, 0.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(1.0, 0.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(1.0, 0.0))
    result.extend(pair(1.0, 0.0))
    result.extend(pair(0.0, 1.0))
    return result^


def observation() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def terminal_sparse_with(
    action_prior: List[Float32], n_iterations: Int
) -> List[Float32]:
    return vbp_channel_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        pair(0.1, 0.9),
        action_prior,
        2,
        n_iterations,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def terminal_sparse() -> List[Float32]:
    return terminal_sparse_with(pair(0.55, 0.45), 2)


def theta_goal() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.1)
    result.append(0.8)
    result.append(0.9)
    result.append(0.2)
    return result^


def test_sparse_terminal_goal_matches_jax() raises:
    var result = terminal_sparse()
    var expected = List[Float32]()
    expected.append(0.5723877549)
    expected.append(0.4276122451)
    expected.append(-0.6020950675)
    expected.append(-0.7933278680)
    expected.append(-0.5936009884)
    expected.append(-0.8037096262)
    expected.append(-0.7556596398)
    expected.append(-0.6343136429)
    expected.append(-0.4655718207)
    expected.append(-0.9882594347)
    assert_equal(len(result), len(expected))
    for index in range(len(expected)):
        assert_close(result[index], expected[index])


def test_dense_terminal_goal_matches_sparse() raises:
    var dense = vbp_channel_planning_dense(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_transition(),
        observation(),
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    var sparse = terminal_sparse()
    for index in range(len(sparse)):
        assert_close(dense[index], sparse[index])


def test_sparse_theta_goal_matches_jax() raises:
    var result = vbp_channel_planning_sparse_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        theta_goal(),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 0.5651110411)
    assert_close(result[1], 0.4348889887)
    assert_close(result[2], -0.6302188039)
    assert_close(result[9], -0.8396785259)


def test_dense_theta_goal_matches_sparse() raises:
    var dense = vbp_channel_planning_dense_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_transition(),
        observation(),
        theta_goal(),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    var sparse = vbp_channel_planning_sparse_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        theta_goal(),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    for index in range(len(sparse)):
        assert_close(dense[index], sparse[index])


def test_action_mask_is_preserved() raises:
    var result = terminal_sparse_with(pair(1.0, 0.0), 2)
    assert_close(result[0], 1.0)
    assert_true(result[1] < 1.0e-6)


def test_action_channels_are_conditional() raises:
    var result = terminal_sparse()
    for time_idx in range(2):
        for state_idx in range(2):
            var offset = 2 + (time_idx * 2 + state_idx) * 2
            assert_close(exp(result[offset]) + exp(result[offset + 1]), 1.0)


def test_additional_iteration_changes_policy() raises:
    var once = terminal_sparse_with(pair(0.55, 0.45), 1)
    var twice = terminal_sparse()
    assert_true(abs(once[0] - twice[0]) > 1.0e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
