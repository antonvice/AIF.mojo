from std.collections import List
from std.sys import argv

from aif_mojo.agent import (
    PLANNER_ACTIVE_INFERENCE,
    PLANNER_DYN_CHANNEL,
    PLANNER_LOOPY_BP,
    PLANNER_LOOPY_VBP,
    PLANNER_NUIJTEN,
    PLANNER_PRECISE_INFO_SEEKING,
    PLANNER_REGION_EXTENDED,
    PLANNER_VBP_CHANNEL,
    dispatch_planner_first_action,
    planning_observation_slice,
)
from aif_mojo.frozen_lake import (
    generate_frozen_lake_goal,
    generate_frozen_lake_observation,
    generate_frozen_lake_transition,
)
from aif_mojo.minigrid import (
    MINIGRID_N_ACTIONS,
    MINIGRID_N_CELL_TYPES,
    MINIGRID_N_DOOR_KEY_STATES,
    MINIGRID_N_ORIENTATIONS,
    generate_minigrid_goal,
    generate_minigrid_observation_tensor,
    generate_minigrid_transition_tensor,
    get_valid_static_configs,
)
from aif_mojo.rocksample import (
    generate_rocksample_goal,
    generate_rocksample_observation,
    generate_rocksample_quality_configs,
    generate_rocksample_transition,
    rocksample_n_actions,
    rocksample_n_events,
)
from aif_mojo.wumpus_world import (
    generate_wumpus_goal,
    generate_wumpus_observation,
    generate_wumpus_transition,
)


def one_hot(length: Int, index: Int) -> List[Float32]:
    var result = List[Float32]()
    for value_idx in range(length):
        if value_idx == index:
            result.append(1.0)
        else:
            result.append(0.0)
    return result^


def uniform(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(1.0 / Float32(length))
    return result^


def reference_action_prior(n_actions: Int) -> List[Float32]:
    """Four unit-weight moves followed by reference cost weight 0.5."""
    var total = 4.0 + 0.5 * Float32(n_actions - 4)
    var result = List[Float32]()
    for action_idx in range(n_actions):
        if action_idx < 4:
            result.append(1.0 / total)
        else:
            result.append(0.5 / total)
    return result^


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


def wumpus_monsters() -> List[Float32]:
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


def emit_policy(
    name: String,
    planner_kind: Int,
    q_state: List[Float32],
    q_static: List[Float32],
    transition: List[Float32],
    observation: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_channels: Int,
    n_obs_types: Int,
    theta_goal: Bool,
):
    var policy = dispatch_planner_first_action(
        planner_kind,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        List[Float32](),
        List[Float32](),
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    for action_idx in range(n_actions):
        print("ACTION", name, action_idx, policy[action_idx])


def emit_all(
    q_state: List[Float32],
    q_static: List[Float32],
    transition: List[Float32],
    observation: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_channels: Int,
    n_obs_types: Int,
    theta_goal: Bool,
):
    emit_policy(
        "loopy-vbp",
        PLANNER_LOOPY_VBP,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    emit_policy(
        "loopy",
        PLANNER_LOOPY_BP,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    emit_policy(
        "region-extended",
        PLANNER_REGION_EXTENDED,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    emit_policy(
        "dyn-channel",
        PLANNER_DYN_CHANNEL,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    emit_policy(
        "nuijten",
        PLANNER_NUIJTEN,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    emit_policy(
        "vbp-channel",
        PLANNER_VBP_CHANNEL,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    emit_policy(
        "precise-info-seeking",
        PLANNER_PRECISE_INFO_SEEKING,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    emit_policy(
        "active-inference",
        PLANNER_ACTIVE_INFERENCE,
        q_state,
        q_static,
        transition,
        observation,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )


def run_frozen_lake():
    var holes = frozen_holes()
    emit_all(
        one_hot(8, 0),
        uniform(2),
        generate_frozen_lake_transition(2, holes, 2, 0.0, 3),
        generate_frozen_lake_observation(2, holes, 2, 0.05, 0.15),
        generate_frozen_lake_goal(2, holes, 2, 3, 1.0, 1.0, 1.0),
        reference_action_prior(5),
        1,
        1,
        0.5,
        8,
        5,
        2,
        12,
        2,
        True,
    )


def run_wumpus_world():
    var pits = wumpus_pits()
    var monsters = wumpus_monsters()
    var gold = wumpus_gold()
    emit_all(
        one_hot(8, 0),
        uniform(2),
        generate_wumpus_transition(2, pits, monsters, 2, 0.0),
        generate_wumpus_observation(2, pits, monsters, gold, 2, 0.1, 0.1),
        generate_wumpus_goal(2, pits, monsters, gold, 2, 1.0, 2.0, 2.0, 1.0),
        reference_action_prior(5),
        1,
        1,
        0.5,
        8,
        5,
        2,
        7,
        2,
        True,
    )


def run_rocksample():
    var positions = List[Int]()
    positions.append(0)
    var qualities = generate_rocksample_quality_configs(1)
    var full_observation = generate_rocksample_observation(
        2, positions, qualities, 1, 2.0, 0.1
    )
    var n_states = 4 * 2 * rocksample_n_events(1)
    emit_all(
        one_hot(n_states, 2),
        uniform(2),
        generate_rocksample_transition(2, positions, 1, 0.0),
        planning_observation_slice(full_observation, 4, 1, 5, 3, n_states, 2),
        generate_rocksample_goal(
            2, positions, qualities, 1, 2.0, 4.0, 2.0, 1.0
        ),
        reference_action_prior(rocksample_n_actions(1)),
        1,
        1,
        0.5,
        n_states,
        rocksample_n_actions(1),
        2,
        1,
        3,
        True,
    )


def run_minigrid():
    var grid_size = 3
    var configs = get_valid_static_configs(grid_size)
    var n_static = len(configs) // 2
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    emit_all(
        one_hot(n_states, 3),
        uniform(n_static),
        generate_minigrid_transition_tensor(grid_size, configs),
        generate_minigrid_observation_tensor(grid_size, configs, 3),
        generate_minigrid_goal(grid_size, 2, 2),
        uniform(MINIGRID_N_ACTIONS),
        1,
        1,
        0.5,
        n_states,
        MINIGRID_N_ACTIONS,
        n_static,
        9,
        MINIGRID_N_CELL_TYPES,
        False,
    )


def main() raises:
    var args = argv()
    if len(args) != 2:
        print("ERROR expected-environment")
    elif args[1] == "frozen_lake":
        run_frozen_lake()
    elif args[1] == "wumpus_world":
        run_wumpus_world()
    elif args[1] == "rocksample":
        run_rocksample()
    elif args[1] == "minigrid":
        run_minigrid()
    else:
        print("ERROR unknown-environment", args[1])
