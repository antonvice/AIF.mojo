from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.nuijten_mp import (
    nuijten_mp_planning_dense,
    nuijten_mp_planning_dense_theta_goal,
    nuijten_mp_planning_sparse,
    nuijten_mp_planning_sparse_theta_goal,
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


def theta_goal() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.1)
    result.append(0.8)
    result.append(0.9)
    result.append(0.2)
    return result^


def test_sparse_terminal_goal_matches_jax_regions() raises:
    var result = nuijten_mp_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    assert_equal(len(result), 26)
    assert_close(result[0], 0.5064566135)
    assert_close(result[1], 0.4935433865)
    assert_close(result[2], 0.4226459265)
    assert_close(result[25], 0.0657880902)
    for time_idx in range(3):
        var total = Float32(0.0)
        for index in range(8):
            total += result[2 + time_idx * 8 + index]
        assert_close(total, 1.0)


def test_dense_terminal_goal_matches_sparse() raises:
    var dense = nuijten_mp_planning_dense(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_transition(),
        observation(),
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    var sparse = nuijten_mp_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    assert_equal(len(dense), 58)
    assert_close(dense[2], -3.3164744377)
    assert_close(dense[33], -4.0132942200)
    for action_idx in range(2):
        assert_close(dense[action_idx], sparse[action_idx])
    for index in range(24):
        assert_close(dense[34 + index], sparse[2 + index])


def test_sparse_theta_goal_matches_jax() raises:
    var result = nuijten_mp_planning_sparse_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        theta_goal(),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 0.4956222773)
    assert_close(result[1], 0.5043777227)
    assert_close(result[2], 0.2730106413)
    assert_close(result[25], 0.0600386113)


def test_dense_theta_goal_matches_sparse() raises:
    var dense = nuijten_mp_planning_dense_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_transition(),
        observation(),
        theta_goal(),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    var sparse = nuijten_mp_planning_sparse_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        theta_goal(),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(dense[2], -5.4293813705)
    assert_close(dense[33], -4.9172549248)
    for action_idx in range(2):
        assert_close(dense[action_idx], sparse[action_idx])
    for index in range(24):
        assert_close(dense[34 + index], sparse[2 + index])


def test_sparse_planner_preserves_action_mask() raises:
    var result = nuijten_mp_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        transition_indices(),
        observation(),
        pair(0.1, 0.9),
        pair(1.0, 0.0),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_true(result[1] < 1.0e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
