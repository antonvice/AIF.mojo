from std.collections import List
from std.sys import argv

from aif_mojo.active_inference import (
    active_inference_planning_dense_until_converged,
)
from aif_mojo.agent import planning_observation_slice
from aif_mojo.dyn_channel_loopy_bp import (
    dyn_channel_loopy_bp_planning_dense_until_converged,
)
from aif_mojo.frozen_lake import (
    generate_frozen_lake_goal,
    generate_frozen_lake_observation,
    generate_frozen_lake_transition,
)
from aif_mojo.loopy_bp import loopy_bp_planning_dense_theta_goal_until_converged
from aif_mojo.rocksample import (
    generate_rocksample_goal,
    generate_rocksample_observation,
    generate_rocksample_quality_configs,
    generate_rocksample_transition,
    rocksample_n_actions,
    rocksample_n_events,
)
from aif_mojo.vbp_channel import vbp_channel_planning_dense_until_converged
from aif_mojo.wumpus_world import (
    generate_wumpus_goal,
    generate_wumpus_observation,
    generate_wumpus_transition,
)
from paper_inputs import (
    paper_frozen_holes,
    paper_rock_positions,
    paper_wumpus_gold,
    paper_wumpus_monsters,
    paper_wumpus_pits,
)


def one_hot(length: Int, index: Int) -> List[Float32]:
    var result = List[Float32](length=length, fill=0.0)
    result[index] = 1.0
    return result^


def uniform(length: Int) -> List[Float32]:
    var result = List[Float32](capacity=length)
    for _ in range(length):
        result.append(1.0 / Float32(length))
    return result^


def action_prior(n_actions: Int, special_weight: Float32) -> List[Float32]:
    var total = Float32(4.0) + special_weight * Float32(n_actions - 4)
    var result = List[Float32](capacity=n_actions)
    for action in range(n_actions):
        if action < 4:
            result.append(1.0 / total)
        else:
            result.append(special_weight / total)
    return result^


def rocksample_action_prior() -> List[Float32]:
    var result = List[Float32]()
    for _ in range(4):
        result.append(1.0 / 6.9)
    for _ in range(3):
        result.append(0.7 / 6.9)
    result.append(0.8 / 6.9)
    return result^


def parse_seed(text: String) -> Int:
    for seed in range(5):
        if text == String(seed):
            return seed
    debug_assert(False, "seed must be 0..4")
    return 0


def parse_damping(text: String) -> Float32:
    if text == "0.25":
        return 0.25
    if text == "0.4":
        return 0.4
    if text == "0.5":
        return 0.5
    if text == "0.6":
        return 0.6
    if text == "0.75":
        return 0.75
    if text == "0.9":
        return 0.9
    if text == "1.0":
        return 1.0
    debug_assert(False, "unsupported Appendix F.5 damping")
    return 1.0


def run_dense(
    environment: String,
    method: String,
    seed: Int,
    damping: Float32,
    q_state: List[Float32],
    q_static: List[Float32],
    transition: List[Float32],
    observation: List[Float32],
    goal: List[Float32],
    prior: List[Float32],
    horizon: Int,
    maximum_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
    adaptive: Bool,
):
    var result: List[Float32]
    var metadata: Int
    if method == "bp":
        result = loopy_bp_planning_dense_theta_goal_until_converged(
            q_state,
            q_static,
            transition,
            goal,
            prior,
            horizon,
            maximum_iterations,
            1.0e-4,
            2,
            n_states,
            n_actions,
            n_static,
        )
        metadata = n_actions
    elif method == "vbp":
        result = vbp_channel_planning_dense_until_converged(
            q_state,
            q_static,
            transition,
            observation,
            goal,
            prior,
            horizon,
            maximum_iterations,
            damping,
            1.0e-4,
            2,
            n_states,
            n_actions,
            n_static,
            n_fov,
            n_obs_types,
            True,
            adaptive,
        )
        metadata = n_actions + horizon * n_states * n_actions
    elif method == "rm-mp":
        result = dyn_channel_loopy_bp_planning_dense_until_converged(
            q_state,
            q_static,
            transition,
            observation,
            goal,
            prior,
            horizon,
            maximum_iterations,
            damping,
            1.0e-4,
            2,
            n_states,
            n_actions,
            n_static,
            n_fov,
            n_obs_types,
            True,
            adaptive,
        )
        metadata = n_actions + horizon * n_states * n_states * n_actions
    else:
        debug_assert(method == "aif-mp", "unknown F.5 method")
        result = active_inference_planning_dense_until_converged(
            q_state,
            q_static,
            transition,
            observation,
            goal,
            prior,
            horizon,
            maximum_iterations,
            damping,
            1.0e-4,
            2,
            n_states,
            n_actions,
            n_static,
            n_fov,
            n_obs_types,
            True,
            adaptive,
        )
        metadata = (
            n_actions
            + horizon * n_states * n_states * n_actions
            + (horizon + 1) * n_fov * n_obs_types * n_states * n_static
        )
    print(
        "RESULT",
        environment,
        method,
        seed,
        damping,
        result[metadata],
        result[metadata + 1],
        result[metadata + 2],
        result[metadata + 3],
    )
    for action in range(n_actions):
        print("ACTION", action, result[action])


def run_frozen(
    method: String, seed: Int, damping: Float32, iterations: Int, adaptive: Bool
):
    comptime grid_size = 4
    comptime n_pos = 16
    comptime n_states = 32
    comptime n_static = 15
    var holes = paper_frozen_holes(seed)
    var full_observation = generate_frozen_lake_observation(
        grid_size, holes, n_static, 0.4, 0.1
    )
    run_dense(
        "frozen-lake",
        method,
        seed,
        damping,
        one_hot(n_states, 0),
        uniform(n_static),
        generate_frozen_lake_transition(grid_size, holes, n_static, 0.1, 15),
        planning_observation_slice(
            full_observation, n_states, n_pos, 48, 2, n_states, n_static
        ),
        generate_frozen_lake_goal(
            grid_size, holes, n_static, 15, 1.0, 2.0, 1.0
        ),
        action_prior(5, 0.1),
        15,
        iterations,
        n_states,
        5,
        n_static,
        n_pos,
        2,
        adaptive,
    )


def run_wumpus(
    method: String, seed: Int, damping: Float32, iterations: Int, adaptive: Bool
):
    comptime grid_size = 5
    comptime n_pos = 25
    comptime n_states = 50
    comptime n_static = 25
    var pits = paper_wumpus_pits(seed)
    var monsters = paper_wumpus_monsters(seed)
    var gold = paper_wumpus_gold(seed)
    run_dense(
        "wumpus-world",
        method,
        seed,
        damping,
        one_hot(n_states, 0),
        uniform(n_static),
        generate_wumpus_transition(grid_size, pits, monsters, n_static, 0.01),
        generate_wumpus_observation(
            grid_size, pits, monsters, gold, n_static, 0.1, 0.05
        ),
        generate_wumpus_goal(
            grid_size, pits, monsters, gold, n_static, 1.0, 1.0, 1.0, 1.0
        ),
        action_prior(5, 0.7),
        8,
        iterations,
        n_states,
        5,
        n_static,
        28,
        2,
        adaptive,
    )


def run_rocksample(
    method: String, seed: Int, damping: Float32, iterations: Int, adaptive: Bool
):
    comptime grid_size = 4
    comptime n_rocks = 3
    comptime n_pos = 16
    comptime n_static = 8
    comptime n_states = 640
    comptime n_actions = 8
    var positions = paper_rock_positions(seed)
    var qualities = generate_rocksample_quality_configs(n_rocks)
    var full_observation = generate_rocksample_observation(
        grid_size, positions, qualities, n_rocks, 2.0, 0.1
    )
    run_dense(
        "rocksample",
        method,
        seed,
        damping,
        one_hot(n_states, 8),
        uniform(n_static),
        generate_rocksample_transition(grid_size, positions, n_rocks, 0.0),
        planning_observation_slice(
            full_observation,
            n_pos,
            n_rocks,
            n_pos + n_rocks,
            3,
            n_states,
            n_static,
        ),
        generate_rocksample_goal(
            grid_size, positions, qualities, n_rocks, 2.0, 4.0, 0.5, 1.0
        ),
        rocksample_action_prior(),
        12,
        iterations,
        n_states,
        n_actions,
        n_static,
        n_rocks,
        3,
        adaptive,
    )


def main() raises:
    var args = argv()
    if len(args) != 7:
        print(
            "ERROR expected environment method damping seed full|smoke fixed|adaptive"
        )
        return
    var environment = args[1]
    var method = args[2]
    var damping = parse_damping(args[3])
    var seed = parse_seed(args[4])
    var iterations = 1000
    if args[5] == "smoke":
        iterations = 2
    var adaptive = args[6] == "adaptive"
    if environment == "frozen-lake":
        run_frozen(method, seed, damping, iterations, adaptive)
    elif environment == "wumpus-world":
        run_wumpus(method, seed, damping, iterations, adaptive)
    else:
        debug_assert(environment == "rocksample", "unknown paper environment")
        run_rocksample(method, seed, damping, iterations, adaptive)
