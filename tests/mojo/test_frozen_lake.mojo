from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.frozen_lake import (
    FROZEN_DOWN,
    FROZEN_LEFT,
    FROZEN_RIGHT,
    FROZEN_SCAN,
    FROZEN_STEP_OBSERVATION_START,
    frozen_lake_is_terminal,
    frozen_lake_next_state,
    frozen_lake_reward,
    frozen_lake_step,
    frozen_lake_state_index,
    frozen_lake_state_position,
    frozen_lake_state_scanned,
    generate_frozen_lake_goal,
    generate_frozen_lake_observation,
    generate_frozen_lake_transition,
    generate_frozen_lake_transition_indices,
    sample_frozen_lake_observation,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def holes() -> List[Float32]:
    # Two valid 2x2 configurations: theta 0 has hole 1, theta 1 has hole 2.
    var result = List[Float32]()
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    return result^


def uniform_draws(value: Float32) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(12):
        result.append(value)
    return result^


def transition_offset(
    new_state: Int, old_state: Int, theta: Int, action: Int
) -> Int:
    return ((new_state * 8 + old_state) * 2 + theta) * 5 + action


def observation_offset(channel: Int, obs: Int, state: Int, theta: Int) -> Int:
    return ((channel * 2 + obs) * 8 + state) * 2 + theta


def test_state_indices_round_trip() raises:
    for scanned in range(2):
        for position in range(4):
            var state = frozen_lake_state_index(position, scanned, 4)
            assert_equal(frozen_lake_state_position(state, 4), position)
            assert_equal(frozen_lake_state_scanned(state, 4), scanned)


def test_transition_tensor_is_stochastic_and_preserves_semantics() raises:
    var transition = generate_frozen_lake_transition(2, holes(), 2, 0.0, 3)
    assert_equal(len(transition), 640)
    for old_state in range(8):
        for theta in range(2):
            for action in range(5):
                var total = Float32(0.0)
                for new_state in range(8):
                    total += transition[
                        transition_offset(new_state, old_state, theta, action)
                    ]
                assert_close(total, 1.0)

    # Hole 1 in theta 0 and the goal are absorbing in both scan modes.
    var absorbing_states = List[Int]()
    absorbing_states.append(1)
    absorbing_states.append(5)
    absorbing_states.append(3)
    absorbing_states.append(7)
    for state in absorbing_states:
        for action in range(5):
            assert_close(
                transition[transition_offset(state, state, 0, action)], 1.0
            )
    # Scan changes mode without moving; right enters position 1.
    assert_close(transition[transition_offset(4, 0, 0, 4)], 1.0)
    assert_close(transition[transition_offset(1, 0, 0, 2)], 1.0)

    var indices = generate_frozen_lake_transition_indices(2, holes(), 2, 3)
    assert_equal(len(indices), 80)
    assert_equal(indices[(0 * 5 + 4) * 2], 4)
    assert_equal(indices[(0 * 5 + 2) * 2], 1)


def test_slip_spreads_only_movement_actions() raises:
    var transition = generate_frozen_lake_transition(2, holes(), 2, 0.3, 3)
    # From top-left intending RIGHT: right=.7, down=.1, wall stays=.2.
    assert_close(transition[transition_offset(1, 0, 0, 2)], 0.7)
    assert_close(transition[transition_offset(2, 0, 0, 2)], 0.1)
    assert_close(transition[transition_offset(0, 0, 0, 2)], 0.2)
    assert_close(transition[transition_offset(4, 0, 0, 4)], 1.0)


def test_observation_tensor_matches_position_and_scan_contracts() raises:
    var observation = generate_frozen_lake_observation(
        2, holes(), 2, 0.05, 0.15
    )
    assert_equal(len(observation), 384)
    for channel in range(12):
        for state in range(8):
            for theta in range(2):
                assert_close(
                    observation[observation_offset(channel, 0, state, theta)]
                    + observation[observation_offset(channel, 1, state, theta)],
                    1.0,
                )
    assert_close(observation[observation_offset(0, 1, 0, 0)], 0.999)
    assert_close(observation[observation_offset(0, 1, 1, 0)], 0.001)
    # Hole-cell channel is noisier at Manhattan distance 2 than distance 1.
    assert_close(observation[observation_offset(9, 1, 0, 0)], 0.875)
    assert_close(observation[observation_offset(9, 1, 2, 0)], 0.8)
    assert_close(observation[observation_offset(9, 1, 4, 0)], 0.999)


def test_goal_and_deterministic_episode_tape() raises:
    var goal = generate_frozen_lake_goal(2, holes(), 2, 3, 1.0, 1.0, 1.0)
    assert_equal(len(goal), 16)
    for theta in range(2):
        var total = Float32(0.0)
        for state in range(8):
            total += goal[state * 2 + theta]
        assert_close(total, 1.0)
        assert_true(goal[3 * 2 + theta] > goal[0 * 2 + theta])
    assert_true(goal[1 * 2] < goal[0])

    var state = Int(0)
    state = frozen_lake_next_state(state, 4, 2, holes(), 0, 3)
    assert_equal(state, 4)
    state = frozen_lake_next_state(state, 1, 2, holes(), 0, 3)
    assert_equal(state, 6)
    state = frozen_lake_next_state(state, 2, 2, holes(), 0, 3)
    assert_equal(state, 7)
    # Goal remains absorbing.
    assert_equal(frozen_lake_next_state(state, 0, 2, holes(), 0, 3), 7)


def test_observation_emission_and_pure_terminal_tape_match_jax() raises:
    var observation = generate_frozen_lake_observation(
        2, holes(), 2, 0.05, 0.15
    )
    var sampled = sample_frozen_lake_observation(
        observation, uniform_draws(0.5), 4, 0, 12, 8, 2
    )
    assert_equal(len(sampled), 12)
    for channel in range(12):
        var expected = Float32(0.0)
        if channel == 4 or channel == 9:
            expected = 1.0
        assert_close(sampled[channel], expected)

    var step = frozen_lake_step(
        0,
        FROZEN_SCAN,
        -1,
        0,
        0,
        3,
        2,
        holes(),
        observation,
        uniform_draws(0.5),
        3,
    )
    assert_equal(len(step), FROZEN_STEP_OBSERVATION_START + 12)
    assert_close(step[0], 4.0)
    assert_close(step[1], 1.0)
    assert_close(step[2], 0.0)
    assert_close(step[3], 0.0)
    assert_close(step[4], 0.0)

    step = frozen_lake_step(
        Int(step[0]),
        FROZEN_DOWN,
        FROZEN_DOWN,
        0,
        Int(step[1]),
        3,
        2,
        holes(),
        observation,
        uniform_draws(0.5),
        3,
    )
    assert_close(step[0], 6.0)
    assert_close(step[1], 2.0)
    step = frozen_lake_step(
        Int(step[0]),
        FROZEN_RIGHT,
        FROZEN_RIGHT,
        0,
        Int(step[1]),
        3,
        2,
        holes(),
        observation,
        uniform_draws(0.5),
        3,
    )
    assert_close(step[0], 7.0)
    assert_close(step[1], 3.0)
    assert_close(step[2], 1.0)
    assert_close(step[3], 1.0)
    # Termination wins when max_steps is reached on the same transition.
    assert_close(step[4], 0.0)
    assert_true(frozen_lake_is_terminal(7, 2, holes(), 0, 3))
    assert_true(frozen_lake_is_terminal(1, 2, holes(), 0, 3))
    assert_close(frozen_lake_reward(7, 2, 3), 1.0)
    assert_close(frozen_lake_reward(1, 2, 3), 0.0)


def test_explicit_slip_observation_draws_and_truncation_match_jax() raises:
    var observation = generate_frozen_lake_observation(
        2, holes(), 2, 0.05, 0.15
    )
    var draws = uniform_draws(0.5)
    draws[8] = 0.1
    draws[9] = 0.85
    draws[10] = 0.04
    draws[11] = 0.2
    # Intended RIGHT realizes DOWN: caller owns the slip draw/RNG adapter.
    var slipped = frozen_lake_step(
        0,
        FROZEN_RIGHT,
        FROZEN_DOWN,
        0,
        0,
        5,
        2,
        holes(),
        observation,
        draws,
        3,
    )
    assert_close(slipped[0], 2.0)
    assert_close(slipped[FROZEN_STEP_OBSERVATION_START + 2], 1.0)
    assert_close(slipped[FROZEN_STEP_OBSERVATION_START + 8], 1.0)
    assert_close(slipped[FROZEN_STEP_OBSERVATION_START + 9], 0.0)
    assert_close(slipped[FROZEN_STEP_OBSERVATION_START + 10], 1.0)
    assert_close(slipped[FROZEN_STEP_OBSERVATION_START + 11], 0.0)

    var boundary = frozen_lake_step(
        0,
        FROZEN_LEFT,
        FROZEN_LEFT,
        0,
        1,
        2,
        2,
        holes(),
        observation,
        uniform_draws(0.5),
        3,
    )
    assert_close(boundary[0], 0.0)
    assert_close(boundary[1], 2.0)
    assert_close(boundary[2], 0.0)
    assert_close(boundary[3], 0.0)
    assert_close(boundary[4], 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
