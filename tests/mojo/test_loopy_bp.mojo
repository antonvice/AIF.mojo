from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_true

from aif_mojo.loopy_bp import (
    backward_messages,
    compute_action_marginals,
    compute_theta_cavities,
    forward_pass,
    loopy_bp_planning_sparse,
    loopy_bp_planning_sparse_theta_goal,
    loopy_bp_planning_dense,
    loopy_bp_planning_dense_theta_goal,
)
from aif_mojo.numerics import LOG_ZERO, safe_log


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def probability_pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def log_pair(a: Float32, b: Float32) -> List[Float32]:
    return probability_pair(safe_log(a), safe_log(b))


def flip_or_stay_reduced(horizon: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(horizon):
        # Shape (new, old, action): action 0 stays, action 1 flips.
        result.append(0.0)
        result.append(LOG_ZERO)
        result.append(LOG_ZERO)
        result.append(0.0)
        result.append(LOG_ZERO)
        result.append(0.0)
        result.append(0.0)
        result.append(LOG_ZERO)
    return result^


def dense_uncertain_transition() -> List[Float32]:
    # Shape (new, old, static, action), matching the sparse planner fixture.
    var result = List[Float32]()
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(1.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(1.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    return result^


def test_forward_and_backward_temporal_messages() raises:
    var reduced = flip_or_stay_reduced(2)
    var action = log_pair(0.7, 0.3)
    var fwd = forward_pass(reduced, log_pair(0.8, 0.2), action, 2, 2, 2)
    assert_close(exp(fwd[0]), 0.8)
    assert_close(exp(fwd[1]), 0.2)
    assert_close(exp(fwd[2]), 0.62)
    assert_close(exp(fwd[3]), 0.38)
    assert_close(exp(fwd[4]), 0.548)
    assert_close(exp(fwd[5]), 0.452)

    var bwd = backward_messages(reduced, log_pair(0.1, 0.9), action, 2, 2, 2)
    assert_close(exp(bwd[0]), 0.436)
    assert_close(exp(bwd[1]), 0.564)
    assert_close(exp(bwd[2]), 0.34)
    assert_close(exp(bwd[3]), 0.66)
    assert_close(exp(bwd[4]), 0.1)
    assert_close(exp(bwd[5]), 0.9)

    var actions = compute_action_marginals(reduced, fwd, bwd, action, 2, 2, 2)
    assert_close(actions[0], 0.61265165)
    assert_close(actions[1], 0.38734835)
    assert_close(actions[2], 0.61265165)
    assert_close(actions[3], 0.38734835)


def test_theta_cavity_excludes_each_factor_message() raises:
    var dyn_messages = List[Float32]()
    dyn_messages.extend(log_pair(0.9, 0.2))
    dyn_messages.extend(log_pair(0.3, 0.8))
    var cavities = compute_theta_cavities(
        log_pair(0.6, 0.4), dyn_messages, 2, 2
    )
    assert_close(exp(cavities[0]), 0.36)
    assert_close(exp(cavities[1]), 0.64)
    assert_close(exp(cavities[2]), 0.87096775)
    assert_close(exp(cavities[3]), 0.12903225)


def test_sparse_planner_matches_jax_terminal_goal_path() raises:
    var transitions = List[Int]()
    transitions.append(0)
    transitions.append(1)
    transitions.append(1)
    transitions.append(0)
    transitions.append(1)
    transitions.append(0)
    transitions.append(0)
    transitions.append(1)
    var result = loopy_bp_planning_sparse(
        probability_pair(1.0, 0.0),
        probability_pair(0.9, 0.1),
        transitions,
        probability_pair(0.01, 0.99),
        probability_pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    assert_close(result[0], 0.10800002)
    assert_close(result[1], 0.892)


def test_sparse_planner_matches_jax_theta_dependent_goal_path() raises:
    var transitions = List[Int]()
    transitions.append(0)
    transitions.append(1)
    transitions.append(1)
    transitions.append(0)
    transitions.append(1)
    transitions.append(0)
    transitions.append(0)
    transitions.append(1)
    var goal = List[Float32]()
    goal.append(0.01)
    goal.append(0.9)
    goal.append(0.99)
    goal.append(0.1)
    var result = loopy_bp_planning_sparse_theta_goal(
        probability_pair(1.0, 0.0),
        probability_pair(0.9, 0.1),
        transitions,
        goal,
        probability_pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    assert_close(result[0], 0.17920004)
    assert_close(result[1], 0.82079995)


def test_dense_planner_matches_sparse_and_jax() raises:
    var result = loopy_bp_planning_dense(
        probability_pair(1.0, 0.0),
        probability_pair(0.9, 0.1),
        dense_uncertain_transition(),
        probability_pair(0.01, 0.99),
        probability_pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    assert_close(result[0], 0.10800002)
    assert_close(result[1], 0.892)


def test_dense_theta_goal_planner_matches_sparse_and_jax() raises:
    var goal = List[Float32]()
    goal.append(0.01)
    goal.append(0.9)
    goal.append(0.99)
    goal.append(0.1)
    var result = loopy_bp_planning_dense_theta_goal(
        probability_pair(1.0, 0.0),
        probability_pair(0.9, 0.1),
        dense_uncertain_transition(),
        goal,
        probability_pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    assert_close(result[0], 0.17920004)
    assert_close(result[1], 0.82079995)


def test_sparse_planner_preserves_action_mask() raises:
    var transitions = List[Int]()
    transitions.append(0)
    transitions.append(1)
    transitions.append(1)
    transitions.append(0)
    transitions.append(1)
    transitions.append(0)
    transitions.append(0)
    transitions.append(1)
    var result = loopy_bp_planning_sparse(
        probability_pair(1.0, 0.0),
        probability_pair(0.9, 0.1),
        transitions,
        probability_pair(0.01, 0.99),
        probability_pair(1.0, 0.0),
        1,
        2,
        2,
        2,
        2,
    )
    assert_close(result[0], 1.0)
    assert_true(result[1] < 1.0e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
