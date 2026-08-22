from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.wumpus_world import (
    WUMPUS_DOWN,
    WUMPUS_RIGHT,
    WUMPUS_SENSE,
    generate_wumpus_goal,
    generate_wumpus_observation,
    generate_wumpus_transition,
    sample_wumpus_observation,
    wumpus_get_neighbors,
    wumpus_is_terminal,
    wumpus_next_state,
    wumpus_reward,
    wumpus_state_index,
    wumpus_state_position,
    wumpus_state_sensed,
    wumpus_step,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def pits() -> List[Float32]:
    # theta 0: pit at 1; theta 1: pit at 2.
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


def wumpus() -> List[Float32]:
    # theta 0: wumpus at 2; theta 1: wumpus at 3.
    var result = List[Float32]()
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    return result^


def gold() -> List[Float32]:
    # theta 0: gold at 3; theta 1: gold at 1.
    var result = List[Float32]()
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    return result^


def transition_offset(
    new_state: Int, old_state: Int, theta: Int, action: Int
) -> Int:
    return ((new_state * 8 + old_state) * 2 + theta) * 5 + action


def observation_offset(channel: Int, obs: Int, state: Int, theta: Int) -> Int:
    return ((channel * 2 + obs) * 8 + state) * 2 + theta


def test_state_index_roundtrip_and_neighbor_order_match_jax() raises:
    for sensed in range(2):
        for position in range(4):
            var state = wumpus_state_index(position, sensed, 4)
            assert_equal(wumpus_state_position(state, 4), position)
            assert_equal(wumpus_state_sensed(state, 4), sensed)
    var neighbors = wumpus_get_neighbors(0, 2)
    assert_equal(len(neighbors), 2)
    assert_equal(neighbors[0], 2)
    assert_equal(neighbors[1], 1)


def test_next_state_and_deterministic_action_tapes() raises:
    # Config 1: SENSE at start, then RIGHT resets the bit and reaches gold.
    var state = Int(0)
    state = wumpus_next_state(state, WUMPUS_SENSE, 2, pits(), wumpus(), 1)
    assert_equal(state, 4)
    state = wumpus_next_state(state, WUMPUS_RIGHT, 2, pits(), wumpus(), 1)
    assert_equal(state, 1)
    assert_close(wumpus_reward(state, 2, pits(), wumpus(), gold(), 1), 1.0)
    assert_true(wumpus_is_terminal(state, 2, pits(), wumpus(), gold(), 1))

    # Config 0: DOWN reaches the wumpus and the death state is absorbing.
    var death = wumpus_next_state(0, WUMPUS_DOWN, 2, pits(), wumpus(), 0)
    assert_equal(death, 2)
    assert_close(wumpus_reward(death, 2, pits(), wumpus(), gold(), 0), -1.0)
    assert_true(wumpus_is_terminal(death, 2, pits(), wumpus(), gold(), 0))
    assert_equal(
        wumpus_next_state(death, WUMPUS_RIGHT, 2, pits(), wumpus(), 0),
        death,
    )


def test_dense_transition_is_complete_and_matches_jax() raises:
    var transition = generate_wumpus_transition(2, pits(), wumpus(), 2, 0.0)
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

    assert_close(transition[transition_offset(4, 0, 0, 4)], 1.0)
    assert_close(transition[transition_offset(1, 0, 0, 2)], 1.0)
    assert_close(transition[transition_offset(2, 4, 0, 1)], 1.0)
    # Pit and wumpus states absorb in both transient-sense modes.
    assert_close(transition[transition_offset(5, 5, 0, 0)], 1.0)
    assert_close(transition[transition_offset(6, 6, 0, 4)], 1.0)


def test_slip_spreads_movement_but_not_sense() raises:
    var transition = generate_wumpus_transition(2, pits(), wumpus(), 2, 0.3)
    # From top-left intending RIGHT: right=.7, down=.1, two walls=.2.
    assert_close(transition[transition_offset(1, 0, 0, 2)], 0.7)
    assert_close(transition[transition_offset(2, 0, 0, 2)], 0.1)
    assert_close(transition[transition_offset(0, 0, 0, 2)], 0.2)
    assert_close(transition[transition_offset(4, 0, 0, 4)], 1.0)


def test_pure_step_covers_goal_and_truncation() raises:
    var observation = generate_wumpus_observation(
        2, pits(), wumpus(), gold(), 2, 0.1, 0.1
    )
    var draws = List[Float32](length=7, fill=0.5)
    var reached = wumpus_step(
        0,
        WUMPUS_RIGHT,
        WUMPUS_RIGHT,
        1,
        0,
        5,
        2,
        pits(),
        wumpus(),
        gold(),
        observation,
        draws,
    )
    assert_equal(Int(reached[0]), 1)
    assert_close(reached[2], 1.0)
    assert_close(reached[3], 1.0)
    assert_close(reached[4], 0.0)
    var truncated = wumpus_step(
        0,
        WUMPUS_SENSE,
        -1,
        0,
        0,
        1,
        2,
        pits(),
        wumpus(),
        gold(),
        observation,
        draws,
    )
    assert_close(truncated[3], 0.0)
    assert_close(truncated[4], 1.0)


def test_observation_tensor_and_explicit_draws_match_jax() raises:
    var observation = generate_wumpus_observation(
        2, pits(), wumpus(), gold(), 2, 0.1, 0.1
    )
    assert_equal(len(observation), 224)
    for channel in range(7):
        for state in range(8):
            for theta in range(2):
                assert_close(
                    observation[observation_offset(channel, 0, state, theta)]
                    + observation[observation_offset(channel, 1, state, theta)],
                    1.0,
                )

    # Feature channels are gated off while idle.
    assert_close(observation[observation_offset(0, 0, 0, 0)], 0.5)
    assert_close(observation[observation_offset(0, 1, 0, 0)], 0.5)
    # At sensed position 0 in config 0 there is a breeze and stench, no glitter.
    assert_close(observation[observation_offset(0, 1, 4, 0)], 0.9)
    assert_close(observation[observation_offset(1, 1, 4, 0)], 0.9)
    assert_close(observation[observation_offset(2, 1, 4, 0)], 0.01)
    # Position channel 0 fires in both modes and is theta-independent.
    assert_close(observation[observation_offset(3, 1, 0, 0)], 0.9)
    assert_close(observation[observation_offset(3, 1, 4, 1)], 0.9)

    var draws = List[Float32]()
    draws.append(0.5)
    draws.append(0.005)
    draws.append(0.02)
    draws.append(0.95)
    draws.append(0.001)
    draws.append(0.5)
    draws.append(0.2)
    var sampled = sample_wumpus_observation(observation, draws, 4, 0, 7, 8, 2)
    var expected = List[Float32]()
    expected.append(1.0)
    expected.append(1.0)
    expected.append(0.0)
    expected.append(0.0)
    expected.append(1.0)
    expected.append(0.0)
    expected.append(0.0)
    for channel in range(7):
        assert_close(sampled[channel], expected[channel])


def test_goal_matches_jax_and_is_flat_over_sense_mode() raises:
    var goal = generate_wumpus_goal(
        2, pits(), wumpus(), gold(), 2, 1.0, 1.0, 1.0, 1.0
    )
    assert_equal(len(goal), 16)
    for theta in range(2):
        var total = Float32(0.0)
        for state in range(8):
            total += goal[state * 2 + theta]
            assert_close(goal[state * 2 + theta], goal[(state % 4) * 2 + theta])
        assert_close(total, 1.0)
    assert_close(goal[0], 0.112257615)
    assert_close(goal[2], 0.041297268)
    assert_close(goal[6], 0.30514786)
    assert_close(goal[3], 0.30514786)
    assert_close(goal[5], 0.041297268)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
