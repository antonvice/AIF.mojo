from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.dyn_channel_loopy_bp import (
    dyn_channel_loopy_bp_planning_dense,
    dyn_channel_loopy_bp_planning_dense_theta_goal,
    dyn_channel_loopy_bp_planning_sparse,
    dyn_channel_loopy_bp_planning_sparse_theta_goal,
)


def assert_close(actual: Float32, expected: Float32) raises:
    assert_true(abs(actual - expected) <= 1.0e-5)


def fixture_inputs() -> List[Float32]:
    """Current state, static state, observation tensor, then action prior."""
    var result = List[Float32]()
    result.append(0.8)
    result.append(0.2)
    result.append(0.6)
    result.append(0.4)
    result.append(0.9)
    result.append(0.2)
    result.append(0.3)
    result.append(0.8)
    result.append(0.1)
    result.append(0.8)
    result.append(0.7)
    result.append(0.2)
    result.append(0.55)
    result.append(0.45)
    return result^


def fixture_transitions() -> List[Int]:
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


def fixture_dense_transition() -> List[Float32]:
    """Deterministic transition in (new, old, theta, action) order."""
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


def zero_values(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def frozen_lake_transition() -> List[Float32]:
    """2x2 Frozen Lake with one hidden hole at position 1 and no slip."""
    comptime n_states = 8
    comptime n_actions = 5
    var result = zero_values(n_states * n_states * n_actions)
    for old_idx in range(n_states):
        var position = old_idx % 4
        var scanned = old_idx // 4
        if position == 1 or position == 3:
            for action_idx in range(n_actions):
                result[
                    (old_idx * n_states + old_idx) * n_actions + action_idx
                ] = 1.0
            continue

        var row = position // 2
        var column = position % 2
        for action_idx in range(4):
            var new_row = row
            var new_column = column
            if action_idx == 0:
                new_column -= 1
            elif action_idx == 1:
                new_row += 1
            elif action_idx == 2:
                new_column += 1
            else:
                new_row -= 1
            var new_position = position
            if (
                new_row >= 0
                and new_row < 2
                and new_column >= 0
                and new_column < 2
            ):
                new_position = new_row * 2 + new_column
            var new_idx = new_position + scanned * 4
            result[
                (new_idx * n_states + old_idx) * n_actions + action_idx
            ] = 1.0
        var scanned_idx = position + 4
        result[(scanned_idx * n_states + old_idx) * n_actions + 4] = 1.0
    return result^


def frozen_lake_transition_indices(transition: List[Float32]) -> List[Int]:
    var result = List[Int]()
    for old_idx in range(8):
        for action_idx in range(5):
            var selected = Int(0)
            for new_idx in range(8):
                var offset = (new_idx * 8 + old_idx) * 5 + action_idx
                if transition[offset] > 0.5:
                    selected = new_idx
            result.append(selected)
    return result^


def frozen_lake_observation() -> List[Float32]:
    """Frozen Lake position and distance-dependent binary hole sensors."""
    var result = zero_values(12 * 2 * 8)
    for state_idx in range(8):
        for channel_idx in range(8):
            var probability = Float32(0.001)
            if channel_idx == state_idx:
                probability = 0.999
            var offset = (channel_idx * 2) * 8 + state_idx
            result[offset] = 1.0 - probability
            result[offset + 8] = probability

    for position in range(4):
        var row = position // 2
        var column = position % 2
        for cell_idx in range(4):
            var cell_row = cell_idx // 2
            var cell_column = cell_idx % 2
            var distance = abs(row - cell_row) + abs(column - cell_column)
            var noise = 0.05 + 0.15 * Float32(distance) / 2.0
            var has_hole = cell_idx == 1
            var probability = noise
            if has_hole:
                probability = 1.0 - noise
            var channel_idx = 8 + cell_idx
            var unscanned_offset = (channel_idx * 2) * 8 + position
            result[unscanned_offset] = 1.0 - probability
            result[unscanned_offset + 8] = probability

            var scanned_probability = Float32(0.001)
            if has_hole:
                scanned_probability = 0.999
            var scanned_offset = (channel_idx * 2) * 8 + position + 4
            result[scanned_offset] = 1.0 - scanned_probability
            result[scanned_offset + 8] = scanned_probability
    return result^


def run_fixture(
    goal: List[Float32], theta_goal: Bool, mask_second_action: Bool
) -> List[Float32]:
    var inputs = fixture_inputs()
    var q_current = List[Float32]()
    q_current.append(inputs[0])
    q_current.append(inputs[1])
    var q_static = List[Float32]()
    q_static.append(inputs[2])
    q_static.append(inputs[3])
    var observation = List[Float32]()
    for index in range(4, 12):
        observation.append(inputs[index])
    var action_prior = List[Float32]()
    if mask_second_action:
        action_prior.append(1.0)
        action_prior.append(0.0)
    else:
        action_prior.append(inputs[12])
        action_prior.append(inputs[13])
    if theta_goal:
        return dyn_channel_loopy_bp_planning_sparse_theta_goal(
            q_current,
            q_static,
            fixture_transitions(),
            observation,
            goal,
            action_prior,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        )
    return dyn_channel_loopy_bp_planning_sparse(
        q_current,
        q_static,
        fixture_transitions(),
        observation,
        goal,
        action_prior,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def run_dense_fixture(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    var inputs = fixture_inputs()
    var q_current = List[Float32]()
    q_current.append(inputs[0])
    q_current.append(inputs[1])
    var q_static = List[Float32]()
    q_static.append(inputs[2])
    q_static.append(inputs[3])
    var observation = List[Float32]()
    for index in range(4, 12):
        observation.append(inputs[index])
    var action_prior = List[Float32]()
    action_prior.append(inputs[12])
    action_prior.append(inputs[13])
    if theta_goal:
        return dyn_channel_loopy_bp_planning_dense_theta_goal(
            q_current,
            q_static,
            fixture_dense_transition(),
            observation,
            goal,
            action_prior,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        )
    return dyn_channel_loopy_bp_planning_dense(
        q_current,
        q_static,
        fixture_dense_transition(),
        observation,
        goal,
        action_prior,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def test_sparse_terminal_goal_planner_matches_jax_after_two_iterations() raises:
    var goal = List[Float32]()
    goal.append(0.1)
    goal.append(0.9)
    var result = run_fixture(goal, False, False)

    var expected = List[Float32]()
    expected.append(0.5735483766)
    expected.append(0.4264516234)
    expected.append(-0.6076425314)
    expected.append(-0.8117039800)
    expected.append(-0.7866522670)
    expected.append(-0.5871681571)
    expected.append(-0.8117038012)
    expected.append(-0.6076425314)
    expected.append(-0.5871682167)
    expected.append(-0.7866523266)
    expected.append(-1.2389082909)
    expected.append(-1.5413109064)
    expected.append(-0.3420683146)
    expected.append(-0.2409259677)
    expected.append(-1.5413109064)
    expected.append(-1.2389080524)
    expected.append(-0.2409260720)
    expected.append(-0.3420683742)
    assert_equal(len(result), len(expected))
    for index in range(len(expected)):
        assert_close(result[index], expected[index])


def test_sparse_theta_goal_planner_matches_jax_after_two_iterations() raises:
    var goal = List[Float32]()
    goal.append(0.1)
    goal.append(0.8)
    goal.append(0.9)
    goal.append(0.2)
    var result = run_fixture(goal, True, False)

    var expected = List[Float32]()
    expected.append(0.5526032448)
    expected.append(0.4473967552)
    expected.append(-0.7956663370)
    expected.append(-0.5368347764)
    expected.append(-0.6001678705)
    expected.append(-0.8785029054)
    expected.append(-0.5368347764)
    expected.append(-0.7956663370)
    expected.append(-0.8785029054)
    expected.append(-0.6001678705)
    expected.append(-0.7936215401)
    expected.append(-0.5222671032)
    expected.append(-0.6018525362)
    expected.append(-0.8993703127)
    expected.append(-0.5222671628)
    expected.append(-0.7936215401)
    expected.append(-0.8993701339)
    expected.append(-0.6018525362)
    assert_equal(len(result), len(expected))
    for index in range(len(expected)):
        assert_close(result[index], expected[index])


def test_sparse_planner_preserves_action_mask() raises:
    var goal = List[Float32]()
    goal.append(0.1)
    goal.append(0.9)
    var result = run_fixture(goal, False, True)
    assert_close(result[0], 1.0)
    assert_true(result[1] < 1.0e-6)


def test_dense_terminal_goal_matches_sparse_and_jax() raises:
    var goal = List[Float32]()
    goal.append(0.1)
    goal.append(0.9)
    var dense = run_dense_fixture(goal, False)
    var sparse = run_fixture(goal, False, False)
    assert_equal(len(dense), len(sparse))
    for index in range(len(sparse)):
        assert_close(dense[index], sparse[index])


def test_dense_theta_goal_matches_sparse_and_jax() raises:
    var goal = List[Float32]()
    goal.append(0.1)
    goal.append(0.8)
    goal.append(0.9)
    goal.append(0.2)
    var dense = run_dense_fixture(goal, True)
    var sparse = run_fixture(goal, True, False)
    assert_equal(len(dense), len(sparse))
    for index in range(len(sparse)):
        assert_close(dense[index], sparse[index])


def test_frozen_lake_fixture_prefers_safe_down_action() raises:
    var current_state = zero_values(8)
    current_state[0] = 1.0
    var static_state = List[Float32]()
    static_state.append(1.0)
    var goal = List[Float32]()
    for _ in range(8):
        goal.append(0.01)
    goal[3] = 1.0
    goal[7] = 1.0
    var action_prior = List[Float32]()
    for _ in range(5):
        action_prior.append(0.2)
    var transition = frozen_lake_transition()
    var observation = frozen_lake_observation()
    var dense = dyn_channel_loopy_bp_planning_dense(
        current_state,
        static_state,
        transition,
        observation,
        goal,
        action_prior,
        2,
        2,
        0.5,
        8,
        5,
        1,
        12,
        2,
    )
    var sparse = dyn_channel_loopy_bp_planning_sparse(
        current_state,
        static_state,
        frozen_lake_transition_indices(transition),
        observation,
        goal,
        action_prior,
        2,
        2,
        0.5,
        8,
        5,
        1,
        12,
        2,
    )
    assert_close(dense[0], 0.0030205285)
    assert_close(dense[1], 0.9879179001)
    for action_idx in range(5):
        assert_close(dense[action_idx], sparse[action_idx])
    for time_idx in range(2):
        for old_idx in range(8):
            for action_idx in range(5):
                var total = Float32(0.0)
                for new_idx in range(8):
                    var offset = (
                        5
                        + ((time_idx * 8 + old_idx) * 8 + new_idx) * 5
                        + action_idx
                    )
                    total += exp(dense[offset])
                assert_close(total, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
