from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.rocksample import (
    ROCK_DOWN,
    ROCK_LEFT,
    ROCK_NO_INFO,
    ROCK_RIGHT,
    ROCK_STEP_OBSERVATION_START,
    ROCK_UP,
    generate_rocksample_goal,
    generate_rocksample_observation,
    generate_rocksample_quality_configs,
    generate_rocksample_transition,
    generate_rocksample_transition_indices,
    rocksample_chebyshev_distance,
    rocksample_event_sample,
    rocksample_event_sense,
    rocksample_is_exit,
    rocksample_is_terminal,
    rocksample_n_actions,
    rocksample_n_events,
    rocksample_next_state,
    rocksample_reward,
    rocksample_sample_action,
    rocksample_sense_accuracy,
    rocksample_sense_action,
    rocksample_state_event,
    rocksample_state_index,
    rocksample_state_mask,
    rocksample_state_position,
    rocksample_step,
    sample_rocksample_observation,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-6
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def rock_positions() -> List[Int]:
    var result = List[Int]()
    result.append(0)
    return result^


def uniform_draws(value: Float32) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(5):
        result.append(value)
    return result^


def transition_offset(
    new_state: Int, old_state: Int, theta: Int, action: Int
) -> Int:
    return ((new_state * 24 + old_state) * 2 + theta) * 6 + action


def observation_offset(
    channel: Int, outcome: Int, state: Int, theta: Int
) -> Int:
    return ((channel * 3 + outcome) * 24 + state) * 2 + theta


def test_helpers_configs_and_state_indices_match_jax() raises:
    assert_equal(rocksample_n_actions(1), 6)
    assert_equal(rocksample_n_events(1), 3)
    assert_equal(rocksample_sense_action(0), 4)
    assert_equal(rocksample_sample_action(1), 5)
    assert_equal(rocksample_event_sense(0), 1)
    assert_equal(rocksample_event_sample(1), 2)
    var qualities = generate_rocksample_quality_configs(1)
    assert_equal(len(qualities), 2)
    assert_close(qualities[0], 0.0)
    assert_close(qualities[1], 1.0)
    for position in range(4):
        for mask in range(2):
            for event in range(3):
                var state = rocksample_state_index(
                    position, mask, event, 4, 2, 3
                )
                assert_equal(rocksample_state_position(state, 4, 2), position)
                assert_equal(rocksample_state_mask(state, 4, 2), mask)
                assert_equal(rocksample_state_event(state, 4, 2), event)
    assert_equal(rocksample_chebyshev_distance(0, 3, 2), 1)
    assert_close(rocksample_sense_accuracy(1, 2.0), 0.8535534143)
    assert_true(rocksample_is_exit(1, 2))
    assert_true(not rocksample_is_exit(2, 2))


def test_next_state_semantics_and_deterministic_action_tape() raises:
    var state = rocksample_state_index(2, 0, 0, 4, 2, 3)
    assert_equal(state, 2)
    state = rocksample_next_state(
        state, ROCK_UP, ROCK_UP, 2, rock_positions(), 1
    )
    assert_equal(state, 0)
    state = rocksample_next_state(state, 4, 0, 2, rock_positions(), 1)
    assert_equal(state, 8)
    state = rocksample_next_state(state, 5, 0, 2, rock_positions(), 1)
    assert_equal(state, 20)
    state = rocksample_next_state(
        state, ROCK_DOWN, ROCK_DOWN, 2, rock_positions(), 1
    )
    assert_equal(state, 6)
    state = rocksample_next_state(
        state, ROCK_RIGHT, ROCK_RIGHT, 2, rock_positions(), 1
    )
    assert_equal(state, 7)
    # Exit states are absorbing and preserve the event/mask encoding.
    assert_equal(
        rocksample_next_state(
            state, ROCK_LEFT, ROCK_LEFT, 2, rock_positions(), 1
        ),
        7,
    )


def test_dense_and_sparse_transition_contracts() raises:
    var transition = generate_rocksample_transition(2, rock_positions(), 1, 0.2)
    assert_equal(len(transition), 6912)
    for old_state in range(24):
        for theta in range(2):
            for action in range(6):
                var total = Float32(0.0)
                for new_state in range(24):
                    total += transition[
                        transition_offset(new_state, old_state, theta, action)
                    ]
                assert_close(total, 1.0)
    # From top-left intending RIGHT: right=.8, down=1/15, two walls=2/15.
    assert_close(transition[transition_offset(1, 0, 0, 2)], 0.8)
    assert_close(transition[transition_offset(2, 0, 0, 2)], 0.0666666701)
    assert_close(transition[transition_offset(0, 0, 0, 2)], 0.1333333403)
    # Dynamics are theta-independent.
    for new_state in range(24):
        for old_state in range(24):
            for action in range(6):
                assert_close(
                    transition[
                        transition_offset(new_state, old_state, 0, action)
                    ],
                    transition[
                        transition_offset(new_state, old_state, 1, action)
                    ],
                )

    var deterministic = generate_rocksample_transition(
        2, rock_positions(), 1, 0.0
    )
    var indices = generate_rocksample_transition_indices(2, rock_positions(), 1)
    assert_equal(len(indices), 288)
    for old_state in range(24):
        for action in range(6):
            for theta in range(2):
                var destination = indices[(old_state * 6 + action) * 2 + theta]
                assert_close(
                    deterministic[
                        transition_offset(destination, old_state, theta, action)
                    ],
                    1.0,
                )


def test_observation_tensor_matches_event_gating_and_jax_values() raises:
    var observation = generate_rocksample_observation(
        2,
        rock_positions(),
        generate_rocksample_quality_configs(1),
        1,
        2.0,
        0.1,
    )
    assert_equal(len(observation), 720)
    for channel in range(5):
        for state in range(24):
            for theta in range(2):
                var total = Float32(0.0)
                for outcome in range(3):
                    total += observation[
                        observation_offset(channel, outcome, state, theta)
                    ]
                assert_close(total, 1.0)
    assert_close(observation[observation_offset(0, 1, 0, 0)], 0.9)
    assert_close(observation[observation_offset(1, 1, 0, 0)], 0.01)
    assert_close(observation[observation_offset(0, 2, 0, 0)], 0.0)
    assert_close(observation[observation_offset(4, ROCK_NO_INFO, 0, 0)], 1.0)
    # SENSE_0 at its rock: quality zero/one is reported with clipped .99.
    assert_close(observation[observation_offset(4, 0, 8, 0)], 0.99)
    assert_close(observation[observation_offset(4, 1, 8, 1)], 0.99)
    # At Chebyshev distance one, alpha = .5 + .5 * 2^(-1/2).
    assert_close(observation[observation_offset(4, 1, 11, 1)], 0.8535534143)
    # SAMPLE at the rock with its mask set is near-deterministic.
    assert_close(observation[observation_offset(4, 1, 20, 1)], 0.999)


def test_goal_matches_jax_and_is_flat_over_event() raises:
    var goal = generate_rocksample_goal(
        2,
        rock_positions(),
        generate_rocksample_quality_configs(1),
        1,
        2.0,
        4.0,
        2.0,
        1.0,
    )
    assert_equal(len(goal), 48)
    assert_close(goal[0 * 2], 0.0195098184)
    assert_close(goal[0 * 2 + 1], 0.0023682227)
    assert_close(goal[1 * 2], 0.1441591531)
    assert_close(goal[4 * 2], 0.0003573348)
    assert_close(goal[5 * 2 + 1], 0.1293005794)
    for theta in range(2):
        var total = Float32(0.0)
        for state in range(24):
            total += goal[state * 2 + theta]
        assert_close(total, 1.0)
        for position in range(4):
            for mask in range(2):
                var base = rocksample_state_index(position, mask, 0, 4, 2, 3)
                for event in range(1, 3):
                    var state = rocksample_state_index(
                        position, mask, event, 4, 2, 3
                    )
                    assert_close(
                        goal[state * 2 + theta], goal[base * 2 + theta]
                    )
    # Under uniform quality belief, blind collection is less preferred.
    assert_true((goal[4 * 2] + goal[4 * 2 + 1]) < (goal[0] + goal[1]))


def test_observation_emission_and_pure_reward_tape_match_jax() raises:
    var qualities = generate_rocksample_quality_configs(1)
    var observation = generate_rocksample_observation(
        2, rock_positions(), qualities, 1, 2.0, 0.1
    )
    var state = rocksample_state_index(2, 0, 0, 4, 2, 3)
    var step = rocksample_step(
        state,
        ROCK_UP,
        ROCK_UP,
        1,
        0,
        5,
        2,
        rock_positions(),
        qualities,
        observation,
        uniform_draws(0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_equal(len(step), ROCK_STEP_OBSERVATION_START + 5)
    assert_close(step[0], 0.0)
    assert_close(step[1], 1.0)
    assert_close(step[2], 0.0)
    assert_close(step[3], 0.0)
    assert_close(step[4], 0.0)
    var expected = List[Float32]()
    expected.append(1.0)
    expected.append(0.0)
    expected.append(0.0)
    expected.append(0.0)
    expected.append(2.0)
    for channel in range(5):
        assert_close(
            step[ROCK_STEP_OBSERVATION_START + channel], expected[channel]
        )

    step = rocksample_step(
        Int(step[0]),
        rocksample_sense_action(0),
        -1,
        1,
        Int(step[1]),
        5,
        2,
        rock_positions(),
        qualities,
        observation,
        uniform_draws(0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_close(step[0], 8.0)
    assert_close(step[1], 2.0)
    assert_close(step[ROCK_STEP_OBSERVATION_START + 4], 1.0)

    step = rocksample_step(
        Int(step[0]),
        rocksample_sample_action(1),
        -1,
        1,
        Int(step[1]),
        5,
        2,
        rock_positions(),
        qualities,
        observation,
        uniform_draws(0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_close(step[0], 20.0)
    assert_close(step[1], 3.0)
    assert_close(step[2], 10.0)
    assert_close(step[3], 0.0)
    assert_close(step[ROCK_STEP_OBSERVATION_START + 4], 1.0)

    step = rocksample_step(
        Int(step[0]),
        ROCK_DOWN,
        ROCK_DOWN,
        1,
        Int(step[1]),
        5,
        2,
        rock_positions(),
        qualities,
        observation,
        uniform_draws(0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_close(step[0], 6.0)
    step = rocksample_step(
        Int(step[0]),
        ROCK_RIGHT,
        ROCK_RIGHT,
        1,
        Int(step[1]),
        5,
        2,
        rock_positions(),
        qualities,
        observation,
        uniform_draws(0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_close(step[0], 7.0)
    assert_close(step[1], 5.0)
    assert_close(step[2], 10.0)
    assert_close(step[3], 1.0)
    # Termination wins at the max_steps boundary.
    assert_close(step[4], 0.0)
    assert_true(rocksample_is_terminal(7, 2))


def test_explicit_movement_and_categorical_draws_match_jax() raises:
    var qualities = generate_rocksample_quality_configs(1)
    var observation = generate_rocksample_observation(
        2, rock_positions(), qualities, 1, 2.0, 0.1
    )
    var draws = uniform_draws(0.5)
    draws[4] = 0.1
    var state = rocksample_state_index(2, 0, 0, 4, 2, 3)
    var sense = rocksample_step(
        state,
        rocksample_sense_action(0),
        -1,
        1,
        0,
        5,
        2,
        rock_positions(),
        qualities,
        observation,
        draws,
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_close(sense[0], 10.0)
    # Quality is one, but draw .1 falls in the .1464466 incorrect interval.
    assert_close(sense[ROCK_STEP_OBSERVATION_START + 4], 0.0)
    var sampled = sample_rocksample_observation(
        observation, draws, 10, 1, 5, 3, 24, 2
    )
    for channel in range(5):
        assert_close(
            sampled[channel], sense[ROCK_STEP_OBSERVATION_START + channel]
        )

    # Intended RIGHT realizes UP; movement also resets the sense event.
    var slipped = rocksample_step(
        Int(sense[0]),
        ROCK_RIGHT,
        ROCK_UP,
        1,
        1,
        5,
        2,
        rock_positions(),
        qualities,
        observation,
        uniform_draws(0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_close(slipped[0], 0.0)
    assert_equal(rocksample_state_event(Int(slipped[0]), 4, 2), 0)

    var bad_next = rocksample_next_state(
        0, rocksample_sample_action(1), 0, 2, rock_positions(), 1
    )
    assert_close(
        rocksample_reward(
            0,
            rocksample_sample_action(1),
            bad_next,
            0,
            2,
            rock_positions(),
            qualities,
            1,
            10.0,
            10.0,
            10.0,
        ),
        -10.0,
    )

    var boundary = rocksample_step(
        state,
        ROCK_LEFT,
        ROCK_LEFT,
        1,
        1,
        2,
        2,
        rock_positions(),
        qualities,
        observation,
        uniform_draws(0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    assert_close(boundary[0], 2.0)
    assert_close(boundary[1], 2.0)
    assert_close(boundary[2], 0.0)
    assert_close(boundary[3], 0.0)
    assert_close(boundary[4], 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
