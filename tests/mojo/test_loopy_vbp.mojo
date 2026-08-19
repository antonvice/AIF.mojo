from std.collections import List
from std.testing import TestSuite, assert_true

from aif_mojo.loopy_bp import compute_reduced_per_t_dense
from aif_mojo.loopy_vbp import (
    backward_pass_vbp,
    compute_dyn_to_theta_vbp,
    forward_pass_vbp,
    loopy_vbp_planning_dense,
    loopy_vbp_planning_dense_theta_goal,
)
from aif_mojo.numerics import safe_log


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def assert_values_close(actual: List[Float32], expected: List[Float32]) raises:
    assert_true(len(actual) == len(expected))
    for index in range(len(actual)):
        assert_close(actual[index], expected[index])


def probability_pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def log_pair(a: Float32, b: Float32) -> List[Float32]:
    return probability_pair(safe_log(a), safe_log(b))


def dense_transition() -> List[Float32]:
    # Shape (new state, old state, theta, action).
    var result = List[Float32]()
    result.append(0.9)
    result.append(0.2)
    result.append(0.4)
    result.append(0.7)
    result.append(0.3)
    result.append(0.8)
    result.append(0.6)
    result.append(0.1)
    result.append(0.1)
    result.append(0.8)
    result.append(0.6)
    result.append(0.3)
    result.append(0.7)
    result.append(0.2)
    result.append(0.4)
    result.append(0.9)
    return result^


def log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
    return result^


def initial_reduced() -> List[Float32]:
    var cavities = List[Float32]()
    cavities.extend(log_pair(0.55, 0.45))
    cavities.extend(log_pair(0.55, 0.45))
    return compute_reduced_per_t_dense(
        log_values(dense_transition()), cavities, 2, 2, 2, 2
    )


def test_vbp_helper_kernels_match_jax_max_and_argmax_semantics() raises:
    var reduced = initial_reduced()
    var backward = backward_pass_vbp(reduced, log_pair(0.15, 0.85), 2, 2, 2)
    var expected_backward = List[Float32]()
    # Value messages, followed by Q values.
    expected_backward.append(-0.69193786)
    expected_backward.append(-0.694358)
    expected_backward.append(-0.68679214)
    expected_backward.append(-0.6995429)
    expected_backward.append(-1.89712)
    expected_backward.append(-0.16251889)
    expected_backward.append(-0.6909183)
    expected_backward.append(-0.6941039)
    expected_backward.append(-0.6939763)
    expected_backward.append(-0.69333845)
    expected_backward.append(-0.97418475)
    expected_backward.append(-0.5933017)
    expected_backward.append(-0.60605246)
    expected_backward.append(-0.6723647)
    assert_values_close(backward, expected_backward)

    var q_values = List[Float32]()
    for index in range(8):
        q_values.append(backward[6 + index])
    var forward = forward_pass_vbp(
        reduced, log_pair(0.65, 0.35), q_values, 2, 2, 2
    )
    var expected_forward = List[Float32]()
    expected_forward.append(-0.43078294)
    expected_forward.append(-1.0498221)
    expected_forward.append(-0.49675834)
    expected_forward.append(-0.9377699)
    expected_forward.append(-0.8464966)
    expected_forward.append(-0.56021726)
    assert_values_close(forward, expected_forward)

    var values = List[Float32]()
    for index in range(6):
        values.append(backward[index])
    var dyn_to_theta = compute_dyn_to_theta_vbp(
        log_values(dense_transition()), forward, values, 2, 2, 2, 2
    )
    var expected_dyn = List[Float32]()
    expected_dyn.append(-0.20417619)
    expected_dyn.append(-0.37852263)
    expected_dyn.append(-0.25379604)
    expected_dyn.append(-0.3439561)
    assert_values_close(dyn_to_theta, expected_dyn)


def test_terminal_goal_matches_jax_and_horizon_changes_policy() raises:
    var short = loopy_vbp_planning_dense(
        probability_pair(0.65, 0.35),
        probability_pair(0.55, 0.45),
        dense_transition(),
        probability_pair(0.15, 0.85),
        1,
        2,
        2,
        2,
        2,
    )
    assert_close(short[0], 0.3471048)
    assert_close(short[1], 0.6528952)

    var longer = loopy_vbp_planning_dense(
        probability_pair(0.65, 0.35),
        probability_pair(0.55, 0.45),
        dense_transition(),
        probability_pair(0.15, 0.85),
        2,
        2,
        2,
        2,
        2,
    )
    assert_close(longer[0], 0.6509737)
    assert_close(longer[1], 0.34902638)
    assert_close(longer[0] + longer[1], 1.0)


def test_theta_goal_matches_jax_and_refines_across_iterations() raises:
    var goal = List[Float32]()
    goal.append(0.15)
    goal.append(0.75)
    goal.append(0.85)
    goal.append(0.25)
    var first = loopy_vbp_planning_dense_theta_goal(
        probability_pair(0.65, 0.35),
        probability_pair(0.55, 0.45),
        dense_transition(),
        goal,
        2,
        1,
        2,
        2,
        2,
    )
    var second = loopy_vbp_planning_dense_theta_goal(
        probability_pair(0.65, 0.35),
        probability_pair(0.55, 0.45),
        dense_transition(),
        goal,
        2,
        2,
        2,
        2,
        2,
    )
    assert_close(first[0], 0.34929493)
    assert_close(first[1], 0.6507051)
    assert_close(second[0], 0.3487406)
    assert_close(second[1], 0.6512594)
    assert_true(abs(first[0] - second[0]) > 1.0e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
