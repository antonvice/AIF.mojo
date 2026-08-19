from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.minigrid import (
    MINIGRID_ACTION_FORWARD,
    MINIGRID_ACTION_PICKUP,
    MINIGRID_ACTION_TOGGLE,
    MINIGRID_ACTION_TURN_LEFT,
    MINIGRID_ACTION_TURN_RIGHT,
    MINIGRID_CELL_DOOR,
    MINIGRID_CELL_KEY,
    MINIGRID_CELL_WALL,
    MINIGRID_N_ACTIONS,
    MINIGRID_N_CELL_TYPES,
    MINIGRID_N_DOOR_KEY_STATES,
    MINIGRID_N_ORIENTATIONS,
    MINIGRID_ORIENTATION_DOWN,
    MINIGRID_ORIENTATION_RIGHT,
    action_to_onehot,
    contains_door,
    contains_key,
    convert_action,
    coords_to_state,
    direction_to_onehot,
    door_position,
    flatten_state_index,
    generate_minigrid_goal,
    generate_minigrid_observation_tensor,
    generate_minigrid_orientation_observation_tensor,
    generate_minigrid_transition_indices,
    generate_minigrid_transition_tensor,
    get_fov,
    get_next_agent_position,
    get_next_door_key_state,
    get_next_orientation,
    get_relative_coords,
    get_valid_static_configs,
    in_fov,
    key_position,
    next_minigrid_state,
    observation_to_onehot,
    relative_to_absolute_coords,
    relative_to_fov_coords,
    soften_minigrid_observation_tensor,
    state_to_coords,
    unflatten_state_index,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def tiny_configs() -> List[Int]:
    # JAX pair (key_pos=1, door_pos=1): key=(0,1), door=(1,1).
    var result = List[Int]()
    result.append(1)
    result.append(1)
    return result^


def transition_index_offset(old_state: Int, action: Int) -> Int:
    return old_state * MINIGRID_N_ACTIONS + action


def dense_transition_offset(new_state: Int, old_state: Int, action: Int) -> Int:
    return (new_state * 108 + old_state) * MINIGRID_N_ACTIONS + action


def observation_offset(fov_x: Int, fov_y: Int, cell: Int, state: Int) -> Int:
    return ((fov_x * 3 + fov_y) * MINIGRID_N_CELL_TYPES + cell) * 108 + state


def test_indices_coordinates_and_valid_configs_match_jax() raises:
    for state in range(9):
        var coords = state_to_coords(state, 3)
        assert_equal(coords_to_state(coords[0], coords[1], 3), state)
    for location in range(9):
        for orientation in range(MINIGRID_N_ORIENTATIONS):
            for door_key_state in range(MINIGRID_N_DOOR_KEY_STATES):
                var flat = flatten_state_index(
                    location,
                    orientation,
                    door_key_state,
                    9,
                    MINIGRID_N_ORIENTATIONS,
                    MINIGRID_N_DOOR_KEY_STATES,
                )
                var unpacked = unflatten_state_index(
                    flat,
                    9,
                    MINIGRID_N_ORIENTATIONS,
                    MINIGRID_N_DOOR_KEY_STATES,
                )
                assert_equal(unpacked[0], location)
                assert_equal(unpacked[1], orientation)
                assert_equal(unpacked[2], door_key_state)

    var configs3 = get_valid_static_configs(3)
    assert_equal(len(configs3), 18)
    assert_equal(configs3[0], 0)
    assert_equal(configs3[1], 0)
    assert_equal(configs3[16], 2)
    assert_equal(configs3[17], 2)
    var configs4 = get_valid_static_configs(4)
    assert_equal(len(configs4), 96)
    assert_equal(configs4[94], 7)
    assert_equal(configs4[95], 7)
    var key = key_position(5, 4)
    var door = door_position(5, 4)
    assert_equal(key[0], 1)
    assert_equal(key[1], 1)
    assert_equal(door[0], 2)
    assert_equal(door[1], 1)


def test_relative_coordinates_fov_coordinates_and_orientation() raises:
    var relative = get_relative_coords(2, 2, MINIGRID_ORIENTATION_RIGHT, 3, 2)
    assert_equal(relative[0], 0)
    assert_equal(relative[1], 1)
    var absolute = relative_to_absolute_coords(
        2, 2, MINIGRID_ORIENTATION_RIGHT, relative[0], relative[1]
    )
    assert_equal(absolute[0], 3)
    assert_equal(absolute[1], 2)
    assert_true(in_fov(-1, 2, 3))
    assert_true(not in_fov(-2, 2, 3))
    var fov_coords = relative_to_fov_coords(0, 1, 3)
    assert_equal(fov_coords[0], 1)
    assert_equal(fov_coords[1], 1)
    assert_equal(
        get_next_orientation(
            MINIGRID_ORIENTATION_RIGHT, MINIGRID_ACTION_TURN_LEFT
        ),
        3,
    )
    assert_equal(
        get_next_orientation(
            MINIGRID_ORIENTATION_RIGHT, MINIGRID_ACTION_TURN_RIGHT
        ),
        MINIGRID_ORIENTATION_DOWN,
    )


def test_key_door_and_movement_rules_match_jax() raises:
    assert_equal(
        get_next_door_key_state(
            0,
            0,
            MINIGRID_ORIENTATION_DOWN,
            0,
            1,
            1,
            1,
            MINIGRID_ACTION_PICKUP,
            0,
        ),
        1,
    )
    assert_equal(
        get_next_door_key_state(
            0,
            1,
            MINIGRID_ORIENTATION_RIGHT,
            0,
            1,
            1,
            1,
            MINIGRID_ACTION_TOGGLE,
            1,
        ),
        2,
    )
    # The key blocks movement while on the ground.
    assert_equal(
        get_next_agent_position(
            0,
            0,
            MINIGRID_ORIENTATION_DOWN,
            1,
            1,
            0,
            1,
            0,
            MINIGRID_ACTION_FORWARD,
            3,
        ),
        0,
    )
    # An open door can be crossed; the rest of its wall column cannot.
    assert_equal(
        get_next_agent_position(
            0,
            1,
            MINIGRID_ORIENTATION_RIGHT,
            1,
            1,
            0,
            1,
            2,
            MINIGRID_ACTION_FORWARD,
            3,
        ),
        4,
    )
    assert_equal(
        get_next_agent_position(
            0,
            0,
            MINIGRID_ORIENTATION_RIGHT,
            1,
            1,
            0,
            1,
            2,
            MINIGRID_ACTION_FORWARD,
            3,
        ),
        0,
    )


def test_fov_and_visibility_match_complete_jax_fixture() raises:
    var fov = get_fov(
        0,
        0,
        MINIGRID_ORIENTATION_DOWN,
        0,
        1,
        1,
        1,
        0,
        3,
        3,
    )
    var expected = List[Int]()
    expected.append(MINIGRID_CELL_WALL)
    expected.append(MINIGRID_CELL_DOOR)
    expected.append(MINIGRID_CELL_WALL)
    expected.append(1)
    expected.append(MINIGRID_CELL_KEY)
    expected.append(1)
    expected.append(MINIGRID_CELL_WALL)
    expected.append(MINIGRID_CELL_WALL)
    expected.append(MINIGRID_CELL_WALL)
    assert_equal(len(fov), len(expected))
    for idx in range(len(fov)):
        assert_equal(fov[idx], expected[idx])

    # A closed door occludes the cells behind it exactly like MiniGrid.
    var occluded = get_fov(
        0,
        1,
        MINIGRID_ORIENTATION_RIGHT,
        0,
        1,
        1,
        1,
        1,
        3,
        3,
    )
    var expected_occluded = List[Int]()
    expected_occluded.append(0)
    expected_occluded.append(2)
    expected_occluded.append(1)
    expected_occluded.append(0)
    expected_occluded.append(4)
    expected_occluded.append(5)
    expected_occluded.append(0)
    expected_occluded.append(2)
    expected_occluded.append(1)
    for idx in range(len(occluded)):
        assert_equal(occluded[idx], expected_occluded[idx])


def test_sparse_transition_indices_match_jax_and_wall_absorption() raises:
    var indices = generate_minigrid_transition_indices(3, tiny_configs())
    assert_equal(len(indices), 108 * MINIGRID_N_ACTIONS)
    for value in indices:
        assert_true(value >= 0 and value < 108)
    # State 3 is location (0,0), DOWN, key on ground.
    assert_equal(indices[transition_index_offset(3, MINIGRID_ACTION_PICKUP)], 4)
    assert_equal(
        indices[transition_index_offset(3, MINIGRID_ACTION_FORWARD)], 3
    )
    # Location (1,0) is a wall cell for this configuration and is absorbing.
    var wall_state = flatten_state_index(3, 0, 0, 9, 4, 3)
    for action in range(MINIGRID_N_ACTIONS):
        assert_equal(
            indices[transition_index_offset(wall_state, action)], wall_state
        )


def test_tiny_dense_transition_expands_sparse_indices() raises:
    var indices = generate_minigrid_transition_indices(3, tiny_configs())
    var dense = generate_minigrid_transition_tensor(3, tiny_configs())
    assert_equal(len(dense), 108 * 108 * MINIGRID_N_ACTIONS)
    for old_state in range(108):
        for action in range(MINIGRID_N_ACTIONS):
            var expected_new = indices[
                transition_index_offset(old_state, action)
            ]
            var total = Float32(0.0)
            for new_state in range(108):
                total += dense[
                    dense_transition_offset(new_state, old_state, action)
                ]
            assert_close(total, 1.0)
            assert_close(
                dense[dense_transition_offset(expected_new, old_state, action)],
                1.0,
            )


def test_hard_observation_tensor_is_onehot_and_matches_jax_image() raises:
    var observation = generate_minigrid_observation_tensor(3, tiny_configs(), 3)
    assert_equal(len(observation), 3 * 3 * 11 * 108)
    for fov_x in range(3):
        for fov_y in range(3):
            for state in range(108):
                var total = Float32(0.0)
                for cell in range(MINIGRID_N_CELL_TYPES):
                    total += observation[
                        observation_offset(fov_x, fov_y, cell, state)
                    ]
                assert_close(total, 1.0)

    # State 3 has JAX FOV [[WALL,DOOR,WALL],[EMPTY,KEY,EMPTY],[WALL,WALL,WALL]].
    var expected = List[Int]()
    expected.append(2)
    expected.append(4)
    expected.append(2)
    expected.append(1)
    expected.append(5)
    expected.append(1)
    expected.append(2)
    expected.append(2)
    expected.append(2)
    for fov_x in range(3):
        for fov_y in range(3):
            var expected_cell = expected[fov_x * 3 + fov_y]
            assert_close(
                observation[observation_offset(fov_x, fov_y, expected_cell, 3)],
                1.0,
            )


def test_soft_observation_tensor_matches_jax_distance_precision() raises:
    var hard = generate_minigrid_observation_tensor(3, tiny_configs(), 3)
    var soft = soften_minigrid_observation_tensor(hard, 3, 0.1, 108, 1)
    # The reference cell (1,1) has zero distance and remains a hard KEY.
    assert_close(soft[observation_offset(1, 1, MINIGRID_CELL_KEY, 3)], 1.0)
    assert_close(soft[observation_offset(1, 1, 0, 3)], 0.0)
    # Corner (0,0) has distance 2: wall=.8+.2/11, every other type=.2/11.
    assert_close(
        soft[observation_offset(0, 0, MINIGRID_CELL_WALL, 3)],
        0.8181818,
    )
    assert_close(soft[observation_offset(0, 0, 0, 3)], 0.018181818)
    var total = Float32(0.0)
    for cell in range(MINIGRID_N_CELL_TYPES):
        total += soft[observation_offset(0, 0, cell, 3)]
    assert_close(total, 1.0)


def test_orientation_goal_and_observation_conversion() raises:
    var orientation = generate_minigrid_orientation_observation_tensor(3)
    assert_equal(len(orientation), MINIGRID_N_ORIENTATIONS * 108)
    assert_close(orientation[MINIGRID_ORIENTATION_DOWN * 108 + 3], 1.0)
    assert_close(orientation[0 * 108 + 3], 0.0)

    var image = List[Int]()
    image.append(2)
    image.append(4)
    image.append(5)
    image.append(1)
    var onehot = observation_to_onehot(image, 2, 2)
    assert_equal(len(onehot), 4 * MINIGRID_N_CELL_TYPES)
    assert_close(onehot[(2 * MINIGRID_N_CELL_TYPES) + MINIGRID_CELL_KEY], 1.0)
    assert_true(contains_key(image))
    assert_true(contains_door(image))
    var direction = direction_to_onehot(MINIGRID_ORIENTATION_DOWN)
    assert_close(direction[MINIGRID_ORIENTATION_DOWN], 1.0)
    var action = action_to_onehot(MINIGRID_ACTION_FORWARD)
    assert_close(action[MINIGRID_ACTION_FORWARD], 1.0)
    assert_equal(
        convert_action(MINIGRID_ACTION_FORWARD), MINIGRID_ACTION_FORWARD
    )

    var goal = generate_minigrid_goal(3, 2, 2)
    assert_equal(len(goal), 108)
    var expected_goal_states = List[Int]()
    expected_goal_states.append(98)
    expected_goal_states.append(101)
    expected_goal_states.append(104)
    expected_goal_states.append(107)
    var goal_total = Float32(0.0)
    for value in goal:
        goal_total += value
    assert_close(goal_total, 1.0)
    for state in expected_goal_states:
        assert_close(goal[state], 0.25)


def test_deterministic_pickup_toggle_goal_tape_matches_jax() raises:
    var state = flatten_state_index(
        0,
        MINIGRID_ORIENTATION_DOWN,
        0,
        9,
        MINIGRID_N_ORIENTATIONS,
        MINIGRID_N_DOOR_KEY_STATES,
    )
    var actions = List[Int]()
    actions.append(MINIGRID_ACTION_PICKUP)
    actions.append(MINIGRID_ACTION_FORWARD)
    actions.append(MINIGRID_ACTION_TURN_LEFT)
    actions.append(MINIGRID_ACTION_TOGGLE)
    actions.append(MINIGRID_ACTION_FORWARD)
    actions.append(MINIGRID_ACTION_FORWARD)
    actions.append(MINIGRID_ACTION_TURN_RIGHT)
    actions.append(MINIGRID_ACTION_FORWARD)
    var expected = List[Int]()
    expected.append(4)
    expected.append(16)
    expected.append(13)
    expected.append(14)
    expected.append(50)
    expected.append(86)
    expected.append(89)
    expected.append(101)
    for idx in range(len(actions)):
        state = next_minigrid_state(state, actions[idx], 3, 1, 1)
        assert_equal(state, expected[idx])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
