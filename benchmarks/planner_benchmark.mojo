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
)
from aif_mojo.active_inference import (
    active_inference_planning_precomputed,
    precompute_obs_channels,
)
from aif_mojo.minigrid import (
    MINIGRID_N_ACTIONS,
    MINIGRID_N_CELL_TYPES,
    MINIGRID_N_DOOR_KEY_STATES,
    MINIGRID_N_ORIENTATIONS,
    generate_minigrid_goal,
    generate_minigrid_observation_tensor,
    generate_minigrid_transition_indices,
    get_valid_static_configs,
)
from aif_mojo.numerics import safe_log
from aif_mojo.sparse_messages import compute_log_base_sparse


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def uniform(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(1.0 / Float32(length))
    return result^


def one_hot(length: Int, index: Int) -> List[Float32]:
    var result = List[Float32]()
    for value_idx in range(length):
        if value_idx == index:
            result.append(1.0)
        else:
            result.append(0.0)
    return result^


def log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
    return result^


def transition() -> List[Float32]:
    """Tiny dense T(new, old, theta, action) fixture."""
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.4, 0.7))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.6, 0.1))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.6, 0.3))
    result.extend(pair(0.7, 0.2))
    result.extend(pair(0.4, 0.9))
    return result^


def observation() -> List[Float32]:
    """Tiny B(channel, outcome, state, theta) fixture."""
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def goal() -> List[Float32]:
    """Theta-dependent C(state, theta) fixture."""
    var result = List[Float32]()
    result.extend(pair(0.15, 0.75))
    result.extend(pair(0.85, 0.25))
    return result^


def plan(planner_kind: Int) -> List[Float32]:
    return dispatch_planner_first_action(
        planner_kind,
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        goal(),
        pair(0.6, 0.4),
        List[Float32](),
        List[Float32](),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
        True,
    )


def emit(name: String, planner_kind: Int):
    var policy = plan(planner_kind)
    print("POLICY", name, policy[0], policy[1])


def emit_minigrid_sparse_active():
    var grid_size = 3
    var configs = get_valid_static_configs(grid_size)
    var n_static = len(configs) // 2
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    var prior = uniform(n_static)
    var local = precompute_obs_channels(
        log_values(generate_minigrid_observation_tensor(grid_size, configs, 3)),
        log_values(prior),
        1,
        0.5,
        n_states,
        n_static,
        9,
        MINIGRID_N_CELL_TYPES,
    )
    var policy = active_inference_planning_precomputed(
        one_hot(n_states, 3),
        prior,
        compute_log_base_sparse(
            generate_minigrid_transition_indices(grid_size, configs),
            log_values(uniform(n_static)),
            n_states,
            MINIGRID_N_ACTIONS,
            n_static,
        ),
        local,
        generate_minigrid_goal(grid_size, 2, 2),
        uniform(MINIGRID_N_ACTIONS),
        1,
        1,
        0.5,
        n_states,
        MINIGRID_N_ACTIONS,
        n_static,
    )
    for action_idx in range(MINIGRID_N_ACTIONS):
        print(
            "SPARSE_ACTION",
            "minigrid-active-sparse",
            action_idx,
            policy[action_idx],
        )


def main() raises:
    var args = argv()
    var requested = String("all")
    if len(args) > 1:
        requested = args[1]
    var matched = False
    if requested == "all" or requested == "loopy-vbp":
        emit("loopy-vbp", PLANNER_LOOPY_VBP)
        matched = True
    if requested == "all" or requested == "loopy":
        emit("loopy", PLANNER_LOOPY_BP)
        matched = True
    if requested == "all" or requested == "region-extended":
        emit("region-extended", PLANNER_REGION_EXTENDED)
        matched = True
    if requested == "all" or requested == "dyn-channel":
        emit("dyn-channel", PLANNER_DYN_CHANNEL)
        matched = True
    if requested == "all" or requested == "nuijten":
        emit("nuijten", PLANNER_NUIJTEN)
        matched = True
    if requested == "all" or requested == "vbp-channel":
        emit("vbp-channel", PLANNER_VBP_CHANNEL)
        matched = True
    if requested == "all" or requested == "precise-info-seeking":
        emit("precise-info-seeking", PLANNER_PRECISE_INFO_SEEKING)
        matched = True
    if requested == "all" or requested == "active-inference":
        emit("active-inference", PLANNER_ACTIVE_INFERENCE)
        matched = True
    if requested == "minigrid-active-sparse":
        emit_minigrid_sparse_active()
        matched = True
    if not matched:
        print("ERROR unknown-method", requested)
