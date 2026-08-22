from std.collections import List
import std.random
from std.sys import argv

from aif_mojo.agent import (
    PLANNER_ACTIVE_INFERENCE,
    PLANNER_DYN_CHANNEL,
    PLANNER_LOOPY_BP,
    PLANNER_NUIJTEN,
    PLANNER_VBP_CHANNEL,
    frozen_lake_agent_reset,
    frozen_lake_agent_step,
    rocksample_agent_reset,
    rocksample_agent_step,
    wumpus_agent_reset,
    wumpus_agent_step,
)
from aif_mojo.frozen_lake import (
    FROZEN_N_ACTIONS,
    FROZEN_SCAN,
    frozen_lake_step,
    generate_frozen_lake_goal,
    generate_frozen_lake_observation,
    generate_frozen_lake_transition,
    sample_frozen_lake_observation,
)
from aif_mojo.rocksample import (
    generate_rocksample_goal,
    generate_rocksample_observation,
    generate_rocksample_quality_configs,
    generate_rocksample_transition,
    rocksample_n_actions,
    rocksample_n_events,
    rocksample_step,
    sample_rocksample_observation,
)
from aif_mojo.wumpus_world import (
    WUMPUS_N_ACTIONS,
    WUMPUS_SENSE,
    generate_wumpus_goal,
    generate_wumpus_observation,
    generate_wumpus_transition,
    sample_wumpus_observation,
    wumpus_step,
)
from paper_inputs import (
    paper_frozen_holes,
    paper_rock_positions,
    paper_wumpus_gold,
    paper_wumpus_monsters,
    paper_wumpus_pits,
)


def planner_kind(method: String) -> Int:
    if method == "bp":
        return PLANNER_LOOPY_BP
    if method == "vbp":
        return PLANNER_VBP_CHANNEL
    if method == "rm-mp":
        return PLANNER_DYN_CHANNEL
    if method == "nuijten-mp":
        return PLANNER_NUIJTEN
    debug_assert(method == "aif-mp", "unknown paper method")
    return PLANNER_ACTIVE_INFERENCE


def uniform_draws(length: Int) -> List[Float32]:
    var result = List[Float32](capacity=length)
    for _ in range(length):
        result.append(Float32(std.random.random_float64()))
    return result^


def uniform_prior(length: Int) -> List[Float32]:
    var result = List[Float32](capacity=length)
    for _ in range(length):
        result.append(1.0 / Float32(length))
    return result^


def weighted_prior(weights: List[Float32]) -> List[Float32]:
    var total = Float32(0.0)
    for value in weights:
        total += value
    var result = List[Float32](capacity=len(weights))
    for value in weights:
        result.append(value / total)
    return result^


def realized_movement(action: Int, slip: Float32) -> Int:
    if action >= 4 or slip <= 0.0:
        return action
    var draw = Float32(std.random.random_float64())
    if draw < 1.0 - slip:
        return action
    var target = Int((draw - (1.0 - slip)) / (slip / 3.0))
    var seen = 0
    for candidate in range(4):
        if candidate != action:
            if seen == min(target, 2):
                return candidate
            seen += 1
    return action


def unpack_beliefs(
    agent_result: List[Float32], n_states: Int, n_static: Int
) -> List[Float32]:
    var result = List[Float32](capacity=n_states + n_static)
    for index in range(n_states + n_static):
        result.append(agent_result[2 + index])
    return result^


def run_frozen(method: String, episodes: Int, smoke: Bool):
    comptime grid = 4
    comptime n_states = 32
    comptime n_static = 15
    comptime n_channels = 48
    var holes = paper_frozen_holes(0)
    var transition = generate_frozen_lake_transition(
        grid, holes, n_static, 0.1, 15
    )
    var observation_tensor = generate_frozen_lake_observation(
        grid, holes, n_static, 0.4, 0.1
    )
    var goal = generate_frozen_lake_goal(
        grid, holes, n_static, 15, 1.0, 2.0, 1.0
    )
    var weights = List[Float32]()
    for _ in range(4):
        weights.append(1.0)
    weights.append(0.1)
    var prior = weighted_prior(weights)
    var iterations = 400
    if smoke:
        iterations = 1
    var damping = Float32(1.0)
    if method == "vbp" or method == "aif-mp":
        damping = 0.9
    elif method == "rm-mp":
        damping = 0.25
    var successes = 0
    var reward_sum = Float32(0.0)
    var step_sum = 0
    for episode in range(episodes):
        std.random.seed(episode)
        var theta = Int(std.random.random_ui64(0, n_static - 1))
        var state = 0
        var beliefs = frozen_lake_agent_reset(n_states, n_static)
        var q_state = List[Float32]()
        var q_static = List[Float32]()
        for index in range(n_states):
            q_state.append(beliefs[index])
        for index in range(n_static):
            q_static.append(beliefs[n_states + index])
        var current_observation = sample_frozen_lake_observation(
            observation_tensor,
            uniform_draws(n_channels),
            state,
            theta,
            n_channels,
            n_states,
            n_static,
        )
        var previous_action = -1
        var steps = 0
        var done = False
        while not done:
            var agent_result = frozen_lake_agent_step(
                planner_kind(method),
                q_state,
                q_static,
                transition,
                observation_tensor,
                current_observation,
                goal,
                prior,
                previous_action,
                15,
                15,
                iterations,
                damping,
                n_states,
                n_static,
                FROZEN_N_ACTIONS,
                n_channels,
                True,
            )
            var action = Int(agent_result[0])
            beliefs = unpack_beliefs(agent_result, n_states, n_static)
            q_state = List[Float32]()
            q_static = List[Float32]()
            for index in range(n_states):
                q_state.append(beliefs[index])
            for index in range(n_static):
                q_static.append(beliefs[n_states + index])
            var realized = -1
            if action != FROZEN_SCAN:
                realized = realized_movement(action, 0.1)
            var env_result = frozen_lake_step(
                state,
                action,
                realized,
                theta,
                steps,
                15,
                grid,
                holes,
                observation_tensor,
                uniform_draws(n_channels),
                15,
            )
            state = Int(env_result[0])
            steps = Int(env_result[1])
            reward_sum += env_result[2]
            if env_result[2] > 0.0:
                successes += 1
            done = env_result[3] > 0.5 or env_result[4] > 0.5
            current_observation = List[Float32]()
            for index in range(n_channels):
                current_observation.append(env_result[5 + index])
            previous_action = action
        step_sum += steps
    print(
        "RESULT frozen-lake",
        method,
        episodes,
        successes,
        reward_sum,
        step_sum,
        0,
        0,
        0,
    )


def run_wumpus(method: String, episodes: Int, smoke: Bool):
    comptime grid = 5
    comptime n_states = 50
    comptime n_static = 25
    comptime n_channels = 28
    var pits = paper_wumpus_pits(0)
    var monsters = paper_wumpus_monsters(0)
    var gold = paper_wumpus_gold(0)
    var transition = generate_wumpus_transition(
        grid, pits, monsters, n_static, 0.01
    )
    var observation_tensor = generate_wumpus_observation(
        grid, pits, monsters, gold, n_static, 0.1, 0.05
    )
    var goal = generate_wumpus_goal(
        grid, pits, monsters, gold, n_static, 1.0, 1.0, 1.0, 1.0
    )
    var weights = List[Float32]()
    for _ in range(4):
        weights.append(1.0)
    weights.append(0.7)
    var prior = weighted_prior(weights)
    var iterations = 150
    if smoke:
        iterations = 1
    var damping = Float32(1.0)
    if method == "vbp":
        damping = 0.9
    elif method == "rm-mp":
        damping = 0.25
    elif method == "aif-mp":
        damping = 0.75
    var successes = 0
    var reward_sum = Float32(0.0)
    var step_sum = 0
    for episode in range(episodes):
        std.random.seed(episode)
        var theta = Int(std.random.random_ui64(0, n_static - 1))
        var state = 0
        var beliefs = wumpus_agent_reset(n_states, n_static)
        var q_state = List[Float32]()
        var q_static = List[Float32]()
        for index in range(n_states):
            q_state.append(beliefs[index])
        for index in range(n_static):
            q_static.append(beliefs[n_states + index])
        var current_observation = sample_wumpus_observation(
            observation_tensor,
            uniform_draws(n_channels),
            state,
            theta,
            n_channels,
            n_states,
            n_static,
        )
        var previous_action = -1
        var steps = 0
        var done = False
        while not done:
            var agent_result = wumpus_agent_step(
                planner_kind(method),
                q_state,
                q_static,
                transition,
                observation_tensor,
                current_observation,
                goal,
                prior,
                previous_action,
                8,
                8,
                iterations,
                damping,
                n_states,
                n_static,
                WUMPUS_N_ACTIONS,
                n_channels,
                2,
                True,
            )
            var action = Int(agent_result[0])
            beliefs = unpack_beliefs(agent_result, n_states, n_static)
            q_state = List[Float32]()
            q_static = List[Float32]()
            for index in range(n_states):
                q_state.append(beliefs[index])
            for index in range(n_static):
                q_static.append(beliefs[n_states + index])
            var realized = -1
            if action != WUMPUS_SENSE:
                realized = realized_movement(action, 0.01)
            var env_result = wumpus_step(
                state,
                action,
                realized,
                theta,
                steps,
                16,
                grid,
                pits,
                monsters,
                gold,
                observation_tensor,
                uniform_draws(n_channels),
            )
            state = Int(env_result[0])
            steps = Int(env_result[1])
            reward_sum += env_result[2]
            if env_result[2] > 0.0:
                successes += 1
            done = env_result[3] > 0.5 or env_result[4] > 0.5
            current_observation = List[Float32]()
            for index in range(n_channels):
                current_observation.append(env_result[5 + index])
            previous_action = action
        step_sum += steps
    print(
        "RESULT wumpus-world",
        method,
        episodes,
        successes,
        reward_sum,
        step_sum,
        0,
        0,
        0,
    )


def run_rocksample(method: String, episodes: Int, smoke: Bool):
    comptime grid = 4
    comptime n_rocks = 3
    comptime n_static = 8
    comptime n_states = 640
    comptime n_channels = 19
    comptime n_actions = 8
    var positions = paper_rock_positions(0)
    var qualities = generate_rocksample_quality_configs(n_rocks)
    var transition = generate_rocksample_transition(
        grid, positions, n_rocks, 0.0
    )
    var observation_tensor = generate_rocksample_observation(
        grid, positions, qualities, n_rocks, 2.0, 0.1
    )
    var goal = generate_rocksample_goal(
        grid, positions, qualities, n_rocks, 2.0, 4.0, 0.5, 1.0
    )
    var weights = List[Float32]()
    for _ in range(4):
        weights.append(1.0)
    for _ in range(3):
        weights.append(0.7)
    weights.append(0.8)
    var prior = weighted_prior(weights)
    var iterations = 100
    if smoke:
        iterations = 1
    var damping = Float32(1.0)
    if method == "vbp" or method == "aif-mp":
        damping = 0.9
    elif method == "rm-mp":
        damping = 0.25
    var successes = 0
    var total_good_collected = 0
    var total_good = 0
    var episodes_with_good = 0
    var reward_sum = Float32(0.0)
    var step_sum = 0
    for episode in range(episodes):
        std.random.seed(episode)
        var theta = Int(std.random.random_ui64(0, n_static - 1))
        var episode_good = 0
        for rock in range(n_rocks):
            if qualities[theta * n_rocks + rock] > 0.5:
                episode_good += 1
        total_good += episode_good
        if episode_good > 0:
            episodes_with_good += 1
        var state = 8
        var beliefs = rocksample_agent_reset(state, n_states, n_static)
        var q_state = List[Float32]()
        var q_static = List[Float32]()
        for index in range(n_states):
            q_state.append(beliefs[index])
        for index in range(n_static):
            q_static.append(beliefs[n_states + index])
        var current_observation = sample_rocksample_observation(
            observation_tensor,
            uniform_draws(n_channels),
            state,
            theta,
            n_channels,
            3,
            n_states,
            n_static,
        )
        var previous_action = -1
        var steps = 0
        var done = False
        while not done:
            var agent_result = rocksample_agent_step(
                planner_kind(method),
                q_state,
                q_static,
                transition,
                observation_tensor,
                current_observation,
                goal,
                prior,
                previous_action,
                12,
                12,
                iterations,
                damping,
                True,
                n_states,
                n_static,
                n_actions,
                16,
                n_rocks,
                3,
            )
            var action = Int(agent_result[0])
            beliefs = unpack_beliefs(agent_result, n_states, n_static)
            q_state = List[Float32]()
            q_static = List[Float32]()
            for index in range(n_states):
                q_state.append(beliefs[index])
            for index in range(n_static):
                q_static.append(beliefs[n_states + index])
            var realized = action
            var env_result = rocksample_step(
                state,
                action,
                realized,
                theta,
                steps,
                25,
                grid,
                positions,
                qualities,
                observation_tensor,
                uniform_draws(n_channels),
                n_rocks,
                2.0,
                3.0,
                1.0,
            )
            state = Int(env_result[0])
            steps = Int(env_result[1])
            reward_sum += env_result[2]
            done = env_result[3] > 0.5 or env_result[4] > 0.5
            if env_result[2] == 2.0:
                total_good_collected += 1
            if done and env_result[3] > 0.5 and env_result[2] > 0.0:
                successes += 1
            current_observation = List[Float32]()
            for index in range(n_channels):
                current_observation.append(env_result[5 + index])
            previous_action = action
        step_sum += steps
    print(
        "RESULT rocksample-authors-code",
        method,
        episodes,
        successes,
        reward_sum,
        step_sum,
        total_good_collected,
        total_good,
        episodes_with_good,
    )


def main() raises:
    var args = argv()
    if len(args) != 4:
        print("ERROR expected environment method full|smoke")
        return
    var environment = args[1]
    var method = args[2]
    var smoke = args[3] == "smoke"
    var episodes = 1000
    if smoke:
        episodes = 1
    if environment == "frozen-lake":
        run_frozen(method, episodes, smoke)
    elif environment == "wumpus-world":
        run_wumpus(method, episodes, smoke)
    else:
        debug_assert(environment == "rocksample", "unknown paper environment")
        run_rocksample(method, episodes, smoke)
