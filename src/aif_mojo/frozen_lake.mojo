from std.collections import List
from std.math import exp


comptime FROZEN_LEFT = 0
comptime FROZEN_DOWN = 1
comptime FROZEN_RIGHT = 2
comptime FROZEN_UP = 3
comptime FROZEN_SCAN = 4
comptime FROZEN_N_ACTIONS = 5
comptime FROZEN_STEP_OBSERVATION_START = 5


def _zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def frozen_lake_state_index(position: Int, scanned: Int, n_pos: Int) -> Int:
    return position + scanned * n_pos


def frozen_lake_state_position(state: Int, n_pos: Int) -> Int:
    return state % n_pos


def frozen_lake_state_scanned(state: Int, n_pos: Int) -> Int:
    return state // n_pos


def frozen_lake_next_state(
    state: Int,
    action: Int,
    grid_size: Int,
    holes: List[Float32],
    theta: Int,
    goal_position: Int,
) -> Int:
    """Apply one supplied action without sampling slip."""
    var n_pos = grid_size * grid_size
    var position = frozen_lake_state_position(state, n_pos)
    var scanned = frozen_lake_state_scanned(state, n_pos)
    if holes[theta * n_pos + position] > 0.5 or position == goal_position:
        return state
    if action == FROZEN_SCAN:
        return frozen_lake_state_index(position, 1, n_pos)

    var row = position // grid_size
    var column = position % grid_size
    var new_row = row
    var new_column = column
    if action == FROZEN_LEFT:
        new_column -= 1
    elif action == FROZEN_DOWN:
        new_row += 1
    elif action == FROZEN_RIGHT:
        new_column += 1
    elif action == FROZEN_UP:
        new_row -= 1
    var new_position = position
    if (
        new_row >= 0
        and new_row < grid_size
        and new_column >= 0
        and new_column < grid_size
    ):
        new_position = new_row * grid_size + new_column
    return frozen_lake_state_index(new_position, scanned, n_pos)


def generate_frozen_lake_transition(
    grid_size: Int,
    holes: List[Float32],
    n_static: Int,
    slip_probability: Float32,
    goal_position: Int,
) -> List[Float32]:
    """Return dense T(new, old, theta, action)."""
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    debug_assert(len(holes) == n_static * n_pos, "hole shape mismatch")
    var result = _zeros(n_states * n_states * n_static * FROZEN_N_ACTIONS)
    for theta in range(n_static):
        for old_state in range(n_states):
            var position = frozen_lake_state_position(old_state, n_pos)
            if (
                holes[theta * n_pos + position] > 0.5
                or position == goal_position
            ):
                for action in range(FROZEN_N_ACTIONS):
                    var offset = (
                        (old_state * n_states + old_state) * n_static + theta
                    ) * FROZEN_N_ACTIONS + action
                    result[offset] = 1.0
                continue

            for intended_action in range(4):
                for actual_action in range(4):
                    var probability = slip_probability / 3.0
                    if actual_action == intended_action:
                        probability = 1.0 - slip_probability
                    if probability == 0.0:
                        continue
                    var next_state = frozen_lake_next_state(
                        old_state,
                        actual_action,
                        grid_size,
                        holes,
                        theta,
                        goal_position,
                    )
                    var offset = (
                        (next_state * n_states + old_state) * n_static + theta
                    ) * FROZEN_N_ACTIONS + intended_action
                    result[offset] += probability

            var scan_state = frozen_lake_state_index(position, 1, n_pos)
            var scan_offset = (
                (scan_state * n_states + old_state) * n_static + theta
            ) * FROZEN_N_ACTIONS + FROZEN_SCAN
            result[scan_offset] = 1.0
    return result^


def generate_frozen_lake_transition_indices(
    grid_size: Int,
    holes: List[Float32],
    n_static: Int,
    goal_position: Int,
) -> List[Int]:
    """Return deterministic sparse indices in (old, action, theta) order."""
    var n_states = 2 * grid_size * grid_size
    var result = List[Int]()
    for old_state in range(n_states):
        for action in range(FROZEN_N_ACTIONS):
            for theta in range(n_static):
                result.append(
                    frozen_lake_next_state(
                        old_state,
                        action,
                        grid_size,
                        holes,
                        theta,
                        goal_position,
                    )
                )
    return result^


def generate_frozen_lake_observation(
    grid_size: Int,
    holes: List[Float32],
    n_static: Int,
    base_noise: Float32,
    noise_range: Float32,
) -> List[Float32]:
    """Return B(channel, binary observation, state, theta)."""
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    var n_channels = n_states + n_pos
    debug_assert(len(holes) == n_static * n_pos, "hole shape mismatch")
    var result = _zeros(n_channels * 2 * n_states * n_static)

    for state in range(n_states):
        for channel in range(n_states):
            var fire_probability = Float32(0.001)
            if channel == state:
                fire_probability = 0.999
            for theta in range(n_static):
                var no_fire_offset = (
                    (channel * 2) * n_states + state
                ) * n_static + theta
                var fire_offset = (
                    (channel * 2 + 1) * n_states + state
                ) * n_static + theta
                result[no_fire_offset] = 1.0 - fire_probability
                result[fire_offset] = fire_probability

    var max_distance = 2 * (grid_size - 1)
    for theta in range(n_static):
        for position in range(n_pos):
            var row = position // grid_size
            var column = position % grid_size
            for cell in range(n_pos):
                var cell_row = cell // grid_size
                var cell_column = cell % grid_size
                var distance = abs(row - cell_row) + abs(column - cell_column)
                var noise = base_noise
                if max_distance > 0:
                    noise += (
                        noise_range * Float32(distance) / Float32(max_distance)
                    )
                var has_hole = holes[theta * n_pos + cell] > 0.5
                var fire_probability = noise
                if has_hole:
                    fire_probability = 1.0 - noise
                fire_probability = min(max(fire_probability, 0.01), 0.99)
                var channel = n_states + cell
                var unscanned_state = position
                var no_fire_offset = (
                    (channel * 2) * n_states + unscanned_state
                ) * n_static + theta
                var fire_offset = (
                    (channel * 2 + 1) * n_states + unscanned_state
                ) * n_static + theta
                result[no_fire_offset] = 1.0 - fire_probability
                result[fire_offset] = fire_probability

                var scanned_probability = Float32(0.001)
                if has_hole:
                    scanned_probability = 0.999
                var scanned_state = position + n_pos
                no_fire_offset = (
                    (channel * 2) * n_states + scanned_state
                ) * n_static + theta
                fire_offset = (
                    (channel * 2 + 1) * n_states + scanned_state
                ) * n_static + theta
                result[no_fire_offset] = 1.0 - scanned_probability
                result[fire_offset] = scanned_probability
    return result^


def generate_frozen_lake_goal(
    grid_size: Int,
    holes: List[Float32],
    n_static: Int,
    goal_position: Int,
    goal_reward: Float32,
    hole_penalty: Float32,
    temperature: Float32,
) -> List[Float32]:
    """Return per-theta softmax preferences in (state, theta) order."""
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    debug_assert(temperature > 0.0, "temperature must be positive")
    var result = _zeros(n_states * n_static)
    for theta in range(n_static):
        var rewards = List[Float32]()
        for state in range(n_states):
            var position = state % n_pos
            var reward = Float32(0.0)
            if position == goal_position:
                reward += goal_reward
            if holes[theta * n_pos + position] > 0.5:
                reward -= hole_penalty
            rewards.append(reward / temperature)
        var maximum = rewards[0]
        for state in range(1, n_states):
            maximum = max(maximum, rewards[state])
        var total = Float32(0.0)
        for state in range(n_states):
            var value = exp(rewards[state] - maximum)
            result[state * n_static + theta] = value
            total += value
        for state in range(n_states):
            result[state * n_static + theta] /= total
    return result^


def sample_frozen_lake_observation(
    observation_tensor: List[Float32],
    uniform_draws: List[Float32],
    state: Int,
    theta: Int,
    n_channels: Int,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Emit B(channel, binary outcome, state, theta) from supplied draws."""
    debug_assert(
        len(observation_tensor) == n_channels * 2 * n_states * n_static,
        "observation shape mismatch",
    )
    debug_assert(len(uniform_draws) == n_channels, "draw shape mismatch")
    debug_assert(state >= 0 and state < n_states, "state out of range")
    debug_assert(theta >= 0 and theta < n_static, "theta out of range")
    var result = List[Float32]()
    for channel in range(n_channels):
        debug_assert(
            uniform_draws[channel] >= 0.0 and uniform_draws[channel] < 1.0,
            "uniform draw out of range",
        )
        var fire_offset = (
            (channel * 2 + 1) * n_states + state
        ) * n_static + theta
        if uniform_draws[channel] < observation_tensor[fire_offset]:
            result.append(1.0)
        else:
            result.append(0.0)
    return result^


def frozen_lake_is_terminal(
    state: Int,
    grid_size: Int,
    holes: List[Float32],
    theta: Int,
    goal_position: Int,
) -> Bool:
    var n_pos = grid_size * grid_size
    debug_assert(len(holes) % n_pos == 0, "hole shape mismatch")
    debug_assert(
        theta >= 0 and theta < len(holes) // n_pos, "theta out of range"
    )
    var position = frozen_lake_state_position(state, n_pos)
    return holes[theta * n_pos + position] > 0.5 or position == goal_position


def frozen_lake_reward(
    state: Int, grid_size: Int, goal_position: Int
) -> Float32:
    if (
        frozen_lake_state_position(state, grid_size * grid_size)
        == goal_position
    ):
        return 1.0
    return 0.0


def frozen_lake_step(
    state: Int,
    action: Int,
    realized_movement_action: Int,
    theta: Int,
    step_count: Int,
    max_steps: Int,
    grid_size: Int,
    holes: List[Float32],
    observation_tensor: List[Float32],
    uniform_draws: List[Float32],
    goal_position: Int,
) -> List[Float32]:
    """Pure simulator step with explicit movement and observation randomness.

    Returns `[next_state, next_step_count, reward, terminated01,
    truncated01, obs...]`. Supported environment state/count values are exactly
    representable in the flat Float32 adapter; exact state logic remains Int.
    """
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    debug_assert(len(holes) % n_pos == 0, "hole shape mismatch")
    var n_static = len(holes) // n_pos
    debug_assert(
        action >= 0 and action < FROZEN_N_ACTIONS, "action out of range"
    )
    debug_assert(step_count >= 0, "step count must be nonnegative")
    debug_assert(max_steps > 0, "max steps must be positive")

    var realized_action = FROZEN_SCAN
    if action != FROZEN_SCAN:
        debug_assert(
            realized_movement_action >= 0 and realized_movement_action < 4,
            "realized movement out of range",
        )
        realized_action = realized_movement_action
    var next_state = frozen_lake_next_state(
        state,
        realized_action,
        grid_size,
        holes,
        theta,
        goal_position,
    )
    var next_step_count = step_count + 1
    var terminated = frozen_lake_is_terminal(
        next_state, grid_size, holes, theta, goal_position
    )
    var truncated = next_step_count >= max_steps and not terminated
    var observation = sample_frozen_lake_observation(
        observation_tensor,
        uniform_draws,
        next_state,
        theta,
        len(uniform_draws),
        n_states,
        n_static,
    )
    var result = List[Float32]()
    result.append(Float32(next_state))
    result.append(Float32(next_step_count))
    result.append(frozen_lake_reward(next_state, grid_size, goal_position))
    if terminated:
        result.append(1.0)
    else:
        result.append(0.0)
    if truncated:
        result.append(1.0)
    else:
        result.append(0.0)
    result.extend(observation^)
    return result^
