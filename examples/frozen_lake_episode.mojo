from std.collections import List

from aif_mojo.agent import (
    PLANNER_LOOPY_BP,
    frozen_lake_agent_reset,
    frozen_lake_agent_step,
)
from aif_mojo.frozen_lake import (
    FROZEN_DOWN,
    FROZEN_LEFT,
    FROZEN_N_ACTIONS,
    FROZEN_RIGHT,
    FROZEN_SCAN,
    FROZEN_STEP_OBSERVATION_START,
    FROZEN_UP,
    frozen_lake_state_position,
    frozen_lake_step,
    generate_frozen_lake_goal,
    generate_frozen_lake_observation,
    generate_frozen_lake_transition,
    sample_frozen_lake_observation,
)


def holes() -> List[Float32]:
    # Two hidden layouts. The live episode uses theta 0: start top-left,
    # hole top-right, goal bottom-right.
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


def uniform_draws(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.5)
    return result^


def action_prior() -> List[Float32]:
    # Movement weight 1, scan weight 0.5, normalized.
    var result = List[Float32]()
    for _ in range(4):
        result.append(1.0 / 4.5)
    result.append(0.5 / 4.5)
    return result^


def action_name(action: Int) -> String:
    if action == FROZEN_LEFT:
        return "LEFT"
    if action == FROZEN_DOWN:
        return "DOWN"
    if action == FROZEN_RIGHT:
        return "RIGHT"
    if action == FROZEN_UP:
        return "UP"
    return "SCAN"


def cell(position: Int, agent_position: Int) -> String:
    if position == agent_position:
        if position == 3:
            return "A/G"
        return " A "
    if position == 1:
        return " H "
    if position == 3:
        return " G "
    return " . "


def render(state: Int):
    var position = frozen_lake_state_position(state, 4)
    print("+-----+-----+")
    print("|", cell(0, position), "|", cell(1, position), "|")
    print("+-----+-----+")
    print("|", cell(2, position), "|", cell(3, position), "|")
    print("+-----+-----+")


def main() raises:
    comptime grid_size = 2
    comptime n_states = 8
    comptime n_static = 2
    comptime n_channels = 12
    comptime goal_position = 3
    comptime true_theta = 0
    comptime max_steps = 4

    var layouts = holes()
    var transition = generate_frozen_lake_transition(
        grid_size, layouts, n_static, 0.0, goal_position
    )
    var observation_model = generate_frozen_lake_observation(
        grid_size, layouts, n_static, 0.05, 0.15
    )
    var goal = generate_frozen_lake_goal(
        grid_size, layouts, n_static, goal_position, 1.0, 1.0, 1.0
    )
    var reset = frozen_lake_agent_reset(n_states, n_static)
    var q_state = List[Float32]()
    var q_static = List[Float32]()
    for state_idx in range(n_states):
        q_state.append(reset[state_idx])
    for theta_idx in range(n_static):
        q_static.append(reset[n_states + theta_idx])

    var state = 0
    var step_count = 0
    var previous_action = -1
    var observation = sample_frozen_lake_observation(
        observation_model,
        uniform_draws(n_channels),
        state,
        true_theta,
        n_channels,
        n_states,
        n_static,
    )

    print("AIF-MOJO — visible Frozen Lake episode")
    print("planner: Loopy BP | hidden layouts: 2 | true layout: theta 0")
    print("legend: A agent, H hole, G goal")
    while step_count < max_steps:
        print()
        print("STEP", step_count)
        render(state)
        var agent = frozen_lake_agent_step(
            PLANNER_LOOPY_BP,
            q_state,
            q_static,
            transition,
            observation_model,
            observation,
            goal,
            action_prior(),
            previous_action,
            max_steps - step_count,
            3,
            2,
            0.5,
            n_states,
            n_static,
            FROZEN_N_ACTIONS,
            n_channels,
            True,
        )
        var action = Int(agent[0])
        print(
            "posterior theta: [",
            agent[2 + n_states],
            ",",
            agent[3 + n_states],
            "]",
        )
        print("selected action:", action_name(action))
        q_state = List[Float32]()
        q_static = List[Float32]()
        for state_idx in range(n_states):
            q_state.append(agent[2 + state_idx])
        for theta_idx in range(n_static):
            q_static.append(agent[2 + n_states + theta_idx])

        var realized_action = action
        if action == FROZEN_SCAN:
            realized_action = -1
        var environment = frozen_lake_step(
            state,
            action,
            realized_action,
            true_theta,
            step_count,
            max_steps,
            grid_size,
            layouts,
            observation_model,
            uniform_draws(n_channels),
            goal_position,
        )
        state = Int(environment[0])
        step_count = Int(environment[1])
        previous_action = action
        observation = List[Float32]()
        for channel in range(n_channels):
            observation.append(
                environment[FROZEN_STEP_OBSERVATION_START + channel]
            )
        if environment[3] > 0.5:
            print()
            print("FINAL")
            render(state)
            if environment[2] > 0.5:
                print("SUCCESS — goal reached in", step_count, "steps")
            else:
                print("TERMINATED — entered a hole")
            return
        if environment[4] > 0.5:
            print("TRUNCATED — step limit reached")
            return
    raise Error("episode did not terminate")
