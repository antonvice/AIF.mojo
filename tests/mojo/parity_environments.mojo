from std.collections import List

from aif_mojo.agent import (
    PLANNER_LOOPY_BP,
    PLANNER_LOOPY_VBP,
    dense_binary_belief_update,
    frozen_lake_loopy_bp_step,
    minigrid_agent_reset,
    minigrid_sparse_active_step,
    previous_action_distribution,
    rocksample_agent_reset,
    rocksample_agent_step,
    wumpus_agent_reset,
    wumpus_agent_step,
)
from aif_mojo.frozen_lake import (
    FROZEN_SCAN,
    frozen_lake_next_state,
    frozen_lake_step,
    generate_frozen_lake_goal,
    generate_frozen_lake_observation,
    generate_frozen_lake_transition,
    generate_frozen_lake_transition_indices,
    sample_frozen_lake_observation,
)
from aif_mojo.wumpus_world import (
    generate_wumpus_goal,
    generate_wumpus_observation,
    generate_wumpus_transition,
    sample_wumpus_observation,
    wumpus_is_terminal,
    wumpus_next_state,
    wumpus_reward,
)
from aif_mojo.rocksample import (
    generate_rocksample_goal,
    generate_rocksample_observation,
    generate_rocksample_quality_configs,
    generate_rocksample_transition,
    generate_rocksample_transition_indices,
    rocksample_next_state,
    rocksample_sense_action,
    rocksample_step,
    rocksample_state_index,
    sample_rocksample_observation,
)
from aif_mojo.minigrid import (
    MINIGRID_ACTION_FORWARD,
    MINIGRID_ACTION_PICKUP,
    MINIGRID_ACTION_TOGGLE,
    MINIGRID_ACTION_TURN_LEFT,
    MINIGRID_ACTION_TURN_RIGHT,
    MINIGRID_N_DOOR_KEY_STATES,
    MINIGRID_N_ACTIONS,
    MINIGRID_N_ORIENTATIONS,
    MINIGRID_ORIENTATION_DOWN,
    flatten_state_index,
    generate_minigrid_goal,
    generate_minigrid_observation_tensor,
    generate_minigrid_orientation_observation_tensor,
    generate_minigrid_transition_indices,
    generate_minigrid_transition_tensor,
    get_fov,
    next_minigrid_state,
    soften_minigrid_observation_tensor,
)


def frozen_holes() -> List[Float32]:
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


def frozen_start_state() -> List[Float32]:
    var result = List[Float32]()
    result.append(1.0)
    for _ in range(7):
        result.append(0.0)
    return result^


def frozen_observation_before_scan() -> List[Float32]:
    var result = List[Float32]()
    result.append(1.0)
    for _ in range(8):
        result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    return result^


def frozen_observation_after_scan() -> List[Float32]:
    var result = List[Float32]()
    for _ in range(4):
        result.append(0.0)
    result.append(1.0)
    for _ in range(4):
        result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    return result^


def uniform_draws(length: Int, value: Float32) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(value)
    return result^


def wumpus_pits() -> List[Float32]:
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


def wumpus_locations() -> List[Float32]:
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


def wumpus_gold() -> List[Float32]:
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


def wumpus_agent_observation() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    return result^


def rock_positions() -> List[Int]:
    var result = List[Int]()
    result.append(0)
    return result^


def rocksample_agent_observation() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(2.0)
    return result^


def minigrid_configs() -> List[Int]:
    var result = List[Int]()
    result.append(1)
    result.append(1)
    return result^


def main():
    var no_slip = generate_frozen_lake_transition(2, frozen_holes(), 2, 0.0, 3)
    for value in no_slip:
        print("frozen_transition", value)

    var slip = generate_frozen_lake_transition(2, frozen_holes(), 2, 0.3, 3)
    for value in slip:
        print("frozen_transition_slip", value)

    var indices = generate_frozen_lake_transition_indices(
        2, frozen_holes(), 2, 3
    )
    for value in indices:
        print("frozen_transition_indices", value)

    var observation = generate_frozen_lake_observation(
        2, frozen_holes(), 2, 0.05, 0.15
    )
    for value in observation:
        print("frozen_observation", value)
    var frozen_sample = sample_frozen_lake_observation(
        observation, uniform_draws(12, 0.5), 4, 0, 12, 8, 2
    )
    for value in frozen_sample:
        print("frozen_observation_sample", value)
    var frozen_simulator_step = frozen_lake_step(
        0,
        FROZEN_SCAN,
        -1,
        0,
        0,
        3,
        2,
        frozen_holes(),
        observation,
        uniform_draws(12, 0.5),
        3,
    )
    for value in frozen_simulator_step:
        print("frozen_simulator_step", value)

    var goal = generate_frozen_lake_goal(2, frozen_holes(), 2, 3, 1.0, 1.0, 1.0)
    for value in goal:
        print("frozen_goal", value)

    var state = Int(0)
    print("frozen_tape", state)
    state = frozen_lake_next_state(state, 4, 2, frozen_holes(), 0, 3)
    print("frozen_tape", state)
    state = frozen_lake_next_state(state, 1, 2, frozen_holes(), 0, 3)
    print("frozen_tape", state)
    state = frozen_lake_next_state(state, 2, 2, frozen_holes(), 0, 3)
    print("frozen_tape", state)

    var static_prior = List[Float32]()
    static_prior.append(0.5)
    static_prior.append(0.5)
    var first_belief = dense_binary_belief_update(
        frozen_start_state(),
        static_prior,
        no_slip,
        observation,
        frozen_observation_before_scan(),
        -1,
        8,
        2,
        5,
        12,
    )
    for value in first_belief:
        print("frozen_agent_belief_1", value)
    var second_state = List[Float32]()
    for index in range(8):
        second_state.append(first_belief[index])
    var second_static = List[Float32]()
    second_static.append(first_belief[8])
    second_static.append(first_belief[9])
    var second_belief = dense_binary_belief_update(
        second_state,
        second_static,
        no_slip,
        observation,
        frozen_observation_after_scan(),
        4,
        8,
        2,
        5,
        12,
    )
    for value in second_belief:
        print("frozen_agent_belief_2", value)
    var action_prior = List[Float32]()
    for _ in range(5):
        action_prior.append(0.2)
    var agent_step = frozen_lake_loopy_bp_step(
        frozen_start_state(),
        static_prior,
        no_slip,
        observation,
        frozen_observation_before_scan(),
        goal,
        action_prior,
        -1,
        2,
        3,
        2,
        8,
        2,
        5,
        12,
    )
    for value in agent_step:
        print("frozen_agent_step", value)

    var wumpus_transition = generate_wumpus_transition(
        2, wumpus_pits(), wumpus_locations(), 2, 0.0
    )
    for value in wumpus_transition:
        print("wumpus_transition", value)
    var wumpus_slip = generate_wumpus_transition(
        2, wumpus_pits(), wumpus_locations(), 2, 0.3
    )
    for value in wumpus_slip:
        print("wumpus_transition_slip", value)
    var wumpus_observation = generate_wumpus_observation(
        2,
        wumpus_pits(),
        wumpus_locations(),
        wumpus_gold(),
        2,
        0.1,
        0.1,
    )
    for value in wumpus_observation:
        print("wumpus_observation", value)
    var wumpus_goal = generate_wumpus_goal(
        2,
        wumpus_pits(),
        wumpus_locations(),
        wumpus_gold(),
        2,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    for value in wumpus_goal:
        print("wumpus_goal", value)
    var wumpus_draws = List[Float32]()
    wumpus_draws.append(0.5)
    wumpus_draws.append(0.005)
    wumpus_draws.append(0.02)
    wumpus_draws.append(0.95)
    wumpus_draws.append(0.001)
    wumpus_draws.append(0.5)
    wumpus_draws.append(0.2)
    var wumpus_sample = sample_wumpus_observation(
        wumpus_observation, wumpus_draws, 4, 0, 7, 8, 2
    )
    for value in wumpus_sample:
        print("wumpus_sample", value)
    var wumpus_state = Int(0)
    print("wumpus_tape", Float32(wumpus_state))
    wumpus_state = wumpus_next_state(
        wumpus_state,
        4,
        2,
        wumpus_pits(),
        wumpus_locations(),
        1,
    )
    print("wumpus_tape", Float32(wumpus_state))
    wumpus_state = wumpus_next_state(
        wumpus_state,
        2,
        2,
        wumpus_pits(),
        wumpus_locations(),
        1,
    )
    print("wumpus_tape", Float32(wumpus_state))
    print(
        "wumpus_tape",
        wumpus_reward(
            wumpus_state,
            2,
            wumpus_pits(),
            wumpus_locations(),
            wumpus_gold(),
            1,
        ),
    )
    if wumpus_is_terminal(
        wumpus_state,
        2,
        wumpus_pits(),
        wumpus_locations(),
        wumpus_gold(),
        1,
    ):
        print("wumpus_tape", 1.0)
    else:
        print("wumpus_tape", 0.0)

    var wumpus_reset = wumpus_agent_reset(8, 2)
    var wumpus_q_state = List[Float32]()
    for index in range(8):
        wumpus_q_state.append(wumpus_reset[index])
    var wumpus_q_static = List[Float32]()
    wumpus_q_static.append(wumpus_reset[8])
    wumpus_q_static.append(wumpus_reset[9])
    var wumpus_agent = wumpus_agent_step(
        PLANNER_LOOPY_BP,
        wumpus_q_state,
        wumpus_q_static,
        wumpus_transition,
        wumpus_observation,
        wumpus_agent_observation(),
        wumpus_goal,
        previous_action_distribution(-1, 5),
        -1,
        2,
        3,
        1,
        1.0,
        8,
        2,
        5,
        7,
        2,
        True,
    )
    for value in wumpus_agent:
        print("wumpus_agent_step", value)

    var rock_qualities = generate_rocksample_quality_configs(1)
    for value in rock_qualities:
        print("rocksample_qualities", value)
    var rock_transition = generate_rocksample_transition(
        2, rock_positions(), 1, 0.2
    )
    for value in rock_transition:
        print("rocksample_transition", value)
    var rock_transition_indices = generate_rocksample_transition_indices(
        2, rock_positions(), 1
    )
    for value in rock_transition_indices:
        print("rocksample_transition_indices", value)
    var rock_observation = generate_rocksample_observation(
        2, rock_positions(), rock_qualities, 1, 2.0, 0.1
    )
    for value in rock_observation:
        print("rocksample_observation", value)
    var rock_sample = sample_rocksample_observation(
        rock_observation, uniform_draws(5, 0.5), 10, 1, 5, 3, 24, 2
    )
    for value in rock_sample:
        print("rocksample_observation_sample", value)
    var rock_simulator_step = rocksample_step(
        rocksample_state_index(2, 0, 0, 4, 2, 3),
        rocksample_sense_action(0),
        -1,
        1,
        0,
        5,
        2,
        rock_positions(),
        rock_qualities,
        rock_observation,
        uniform_draws(5, 0.5),
        1,
        10.0,
        10.0,
        10.0,
    )
    for value in rock_simulator_step:
        print("rocksample_simulator_step", value)
    var rock_goal = generate_rocksample_goal(
        2, rock_positions(), rock_qualities, 1, 2.0, 4.0, 2.0, 1.0
    )
    for value in rock_goal:
        print("rocksample_goal", value)
    var rock_state = rocksample_state_index(2, 0, 0, 4, 2, 3)
    print("rocksample_tape", rock_state)
    rock_state = rocksample_next_state(rock_state, 3, 3, 2, rock_positions(), 1)
    print("rocksample_tape", rock_state)
    rock_state = rocksample_next_state(rock_state, 4, 0, 2, rock_positions(), 1)
    print("rocksample_tape", rock_state)
    rock_state = rocksample_next_state(rock_state, 5, 0, 2, rock_positions(), 1)
    print("rocksample_tape", rock_state)
    rock_state = rocksample_next_state(rock_state, 1, 1, 2, rock_positions(), 1)
    print("rocksample_tape", rock_state)
    rock_state = rocksample_next_state(rock_state, 2, 2, 2, rock_positions(), 1)
    print("rocksample_tape", rock_state)

    var rock_deterministic = generate_rocksample_transition(
        2, rock_positions(), 1, 0.0
    )
    var rock_reset = rocksample_agent_reset(2, 24, 2)
    var rock_q_state = List[Float32]()
    for index in range(24):
        rock_q_state.append(rock_reset[index])
    var rock_q_static = List[Float32]()
    rock_q_static.append(rock_reset[24])
    rock_q_static.append(rock_reset[25])
    var rock_agent_first = rocksample_agent_step(
        PLANNER_LOOPY_VBP,
        rock_q_state,
        rock_q_static,
        rock_deterministic,
        rock_observation,
        rocksample_agent_observation(),
        rock_goal,
        previous_action_distribution(-1, 6),
        -1,
        2,
        3,
        1,
        1.0,
        True,
        24,
        2,
        6,
        4,
        1,
        3,
    )
    for value in rock_agent_first:
        print("rocksample_agent_step_1", value)
    var rock_second_state = List[Float32]()
    for index in range(24):
        rock_second_state.append(rock_agent_first[2 + index])
    var rock_second_static = List[Float32]()
    rock_second_static.append(rock_agent_first[26])
    rock_second_static.append(rock_agent_first[27])
    var rock_agent_second = rocksample_agent_step(
        PLANNER_LOOPY_VBP,
        rock_second_state,
        rock_second_static,
        rock_deterministic,
        rock_observation,
        rocksample_agent_observation(),
        rock_goal,
        previous_action_distribution(0, 6),
        0,
        1,
        3,
        1,
        1.0,
        True,
        24,
        2,
        6,
        4,
        1,
        3,
    )
    for value in rock_agent_second:
        print("rocksample_agent_step_2", value)

    var minigrid_indices = generate_minigrid_transition_indices(
        3, minigrid_configs()
    )
    for value in minigrid_indices:
        print("minigrid_transition_indices", value)
    var minigrid_transition = generate_minigrid_transition_tensor(
        3, minigrid_configs()
    )
    for value in minigrid_transition:
        print("minigrid_transition", value)
    var minigrid_hard = generate_minigrid_observation_tensor(
        3, minigrid_configs(), 3
    )
    for value in minigrid_hard:
        print("minigrid_observation", value)
    var minigrid_soft = soften_minigrid_observation_tensor(
        minigrid_hard, 3, 0.1, 108, 1
    )
    for value in minigrid_soft:
        print("minigrid_observation_soft", value)
    var minigrid_orientation = generate_minigrid_orientation_observation_tensor(
        3
    )
    for value in minigrid_orientation:
        print("minigrid_orientation", value)
    var minigrid_goal = generate_minigrid_goal(3, 2, 2)
    for value in minigrid_goal:
        print("minigrid_goal", value)
    var minigrid_reset = minigrid_agent_reset(3, 1)
    var minigrid_q_state = List[Float32]()
    for index in range(108):
        minigrid_q_state.append(minigrid_reset[index])
    var minigrid_q_static = List[Float32]()
    minigrid_q_static.append(minigrid_reset[108])
    var minigrid_image = get_fov(
        0, 0, MINIGRID_ORIENTATION_DOWN, 0, 1, 1, 1, 0, 3, 3
    )
    var minigrid_agent = minigrid_sparse_active_step(
        minigrid_q_state,
        minigrid_q_static,
        minigrid_indices,
        minigrid_hard,
        minigrid_orientation,
        minigrid_image,
        MINIGRID_ORIENTATION_DOWN,
        minigrid_goal,
        previous_action_distribution(-1, MINIGRID_N_ACTIONS),
        0,
        1,
        3,
        1,
        1,
        1.0,
        3,
        3,
        1,
        False,
    )
    for value in minigrid_agent:
        print("minigrid_agent_step", value)
    var minigrid_state = flatten_state_index(
        0,
        MINIGRID_ORIENTATION_DOWN,
        0,
        9,
        MINIGRID_N_ORIENTATIONS,
        MINIGRID_N_DOOR_KEY_STATES,
    )
    print("minigrid_tape", minigrid_state)
    var minigrid_actions = List[Int]()
    minigrid_actions.append(MINIGRID_ACTION_PICKUP)
    minigrid_actions.append(MINIGRID_ACTION_FORWARD)
    minigrid_actions.append(MINIGRID_ACTION_TURN_LEFT)
    minigrid_actions.append(MINIGRID_ACTION_TOGGLE)
    minigrid_actions.append(MINIGRID_ACTION_FORWARD)
    minigrid_actions.append(MINIGRID_ACTION_FORWARD)
    minigrid_actions.append(MINIGRID_ACTION_TURN_RIGHT)
    minigrid_actions.append(MINIGRID_ACTION_FORWARD)
    for action in minigrid_actions:
        minigrid_state = next_minigrid_state(minigrid_state, action, 3, 1, 1)
        print("minigrid_tape", minigrid_state)
