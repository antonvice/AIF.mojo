from std.collections import List
from std.math import exp


comptime WUMPUS_LEFT = 0
comptime WUMPUS_DOWN = 1
comptime WUMPUS_RIGHT = 2
comptime WUMPUS_UP = 3
comptime WUMPUS_SENSE = 4
comptime WUMPUS_N_ACTIONS = 5
comptime WUMPUS_N_FEATURE_CHANNELS = 3
comptime WUMPUS_N_OBS_TYPES = 2


def _zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def wumpus_state_index(position: Int, sensed: Int, n_pos: Int) -> Int:
    return position + sensed * n_pos


def wumpus_state_position(state: Int, n_pos: Int) -> Int:
    return state % n_pos


def wumpus_state_sensed(state: Int, n_pos: Int) -> Int:
    return state // n_pos


def wumpus_get_neighbors(position: Int, grid_size: Int) -> List[Int]:
    """Return orthogonal neighbors in JAX LEFT, DOWN, RIGHT, UP order."""
    var row = position // grid_size
    var column = position % grid_size
    var result = List[Int]()
    for action in range(4):
        var new_row = row
        var new_column = column
        if action == WUMPUS_LEFT:
            new_column -= 1
        elif action == WUMPUS_DOWN:
            new_row += 1
        elif action == WUMPUS_RIGHT:
            new_column += 1
        else:
            new_row -= 1
        if (
            new_row >= 0
            and new_row < grid_size
            and new_column >= 0
            and new_column < grid_size
        ):
            result.append(new_row * grid_size + new_column)
    return result^


def _is_hazard(
    position: Int,
    theta: Int,
    n_pos: Int,
    pits: List[Float32],
    wumpus: List[Float32],
) -> Bool:
    return (
        pits[theta * n_pos + position] > 0.5
        or wumpus[theta * n_pos + position] > 0.5
    )


def wumpus_next_state(
    state: Int,
    action: Int,
    grid_size: Int,
    pits: List[Float32],
    wumpus: List[Float32],
    theta: Int,
) -> Int:
    """Apply one explicit realized action without native RNG.

    Movement resets the transient sensed bit. Hazard states are absorbing,
    matching the transition tensor. For slip, callers supply the realized
    movement action rather than the originally intended action.
    """
    var n_pos = grid_size * grid_size
    debug_assert(len(pits) == len(wumpus), "hazard shape mismatch")
    debug_assert(len(pits) % n_pos == 0, "hazard shape mismatch")
    debug_assert(action >= 0 and action < WUMPUS_N_ACTIONS, "invalid action")
    var position = wumpus_state_position(state, n_pos)
    if _is_hazard(position, theta, n_pos, pits, wumpus):
        return state
    if action == WUMPUS_SENSE:
        return wumpus_state_index(position, 1, n_pos)

    var row = position // grid_size
    var column = position % grid_size
    var new_row = row
    var new_column = column
    if action == WUMPUS_LEFT:
        new_column -= 1
    elif action == WUMPUS_DOWN:
        new_row += 1
    elif action == WUMPUS_RIGHT:
        new_column += 1
    else:
        new_row -= 1
    var new_position = position
    if (
        new_row >= 0
        and new_row < grid_size
        and new_column >= 0
        and new_column < grid_size
    ):
        new_position = new_row * grid_size + new_column
    return wumpus_state_index(new_position, 0, n_pos)


def generate_wumpus_transition(
    grid_size: Int,
    pits: List[Float32],
    wumpus: List[Float32],
    n_static: Int,
    slip_probability: Float32,
) -> List[Float32]:
    """Return complete dense T(new, old, theta, action)."""
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    debug_assert(len(pits) == n_static * n_pos, "pit shape mismatch")
    debug_assert(len(wumpus) == n_static * n_pos, "wumpus shape mismatch")
    var result = _zeros(n_states * n_states * n_static * WUMPUS_N_ACTIONS)
    for theta in range(n_static):
        for old_state in range(n_states):
            var position = wumpus_state_position(old_state, n_pos)
            if _is_hazard(position, theta, n_pos, pits, wumpus):
                for action in range(WUMPUS_N_ACTIONS):
                    var offset = (
                        (old_state * n_states + old_state) * n_static + theta
                    ) * WUMPUS_N_ACTIONS + action
                    result[offset] = 1.0
                continue

            for intended_action in range(4):
                for actual_action in range(4):
                    var probability = slip_probability / 3.0
                    if actual_action == intended_action:
                        probability = 1.0 - slip_probability
                    if probability == 0.0:
                        continue
                    var new_state = wumpus_next_state(
                        old_state,
                        actual_action,
                        grid_size,
                        pits,
                        wumpus,
                        theta,
                    )
                    var offset = (
                        (new_state * n_states + old_state) * n_static + theta
                    ) * WUMPUS_N_ACTIONS + intended_action
                    result[offset] += probability

            var sense_state = wumpus_state_index(position, 1, n_pos)
            var sense_offset = (
                (sense_state * n_states + old_state) * n_static + theta
            ) * WUMPUS_N_ACTIONS + WUMPUS_SENSE
            result[sense_offset] = 1.0
    return result^


def generate_wumpus_observation(
    grid_size: Int,
    pits: List[Float32],
    wumpus: List[Float32],
    gold: List[Float32],
    n_static: Int,
    observation_noise: Float32,
    position_noise: Float32,
) -> List[Float32]:
    """Return B(channel, binary observation, state, theta)."""
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    var n_channels = WUMPUS_N_FEATURE_CHANNELS + n_pos
    debug_assert(len(pits) == n_static * n_pos, "pit shape mismatch")
    debug_assert(len(wumpus) == n_static * n_pos, "wumpus shape mismatch")
    debug_assert(len(gold) == n_static * n_pos, "gold shape mismatch")
    var result = _zeros(n_channels * WUMPUS_N_OBS_TYPES * n_states * n_static)
    var true_positive = min(max(1.0 - observation_noise, 0.01), 0.99)
    var false_positive = min(max(observation_noise * 0.1, 0.01), 0.99)
    var position_true_positive = min(max(1.0 - position_noise, 0.01), 0.999)
    var position_false_positive = min(max(position_noise * 0.1, 0.001), 0.99)

    for theta in range(n_static):
        for position in range(n_pos):
            var neighbors = wumpus_get_neighbors(position, grid_size)
            var has_breeze = False
            var has_stench = False
            for neighbor in neighbors:
                if pits[theta * n_pos + neighbor] > 0.5:
                    has_breeze = True
                if wumpus[theta * n_pos + neighbor] > 0.5:
                    has_stench = True
            var has_glitter = gold[theta * n_pos + position] > 0.5

            var idle_state = wumpus_state_index(position, 0, n_pos)
            for channel in range(WUMPUS_N_FEATURE_CHANNELS):
                var no_fire_offset = (
                    (channel * 2) * n_states + idle_state
                ) * n_static + theta
                var fire_offset = (
                    (channel * 2 + 1) * n_states + idle_state
                ) * n_static + theta
                result[no_fire_offset] = 0.5
                result[fire_offset] = 0.5

            var sensed_state = wumpus_state_index(position, 1, n_pos)
            for channel in range(WUMPUS_N_FEATURE_CHANNELS):
                var feature_present = has_glitter
                if channel == 0:
                    feature_present = has_breeze
                elif channel == 1:
                    feature_present = has_stench
                var probability = false_positive
                if feature_present:
                    probability = true_positive
                var no_fire_offset = (
                    (channel * 2) * n_states + sensed_state
                ) * n_static + theta
                var fire_offset = (
                    (channel * 2 + 1) * n_states + sensed_state
                ) * n_static + theta
                result[no_fire_offset] = 1.0 - probability
                result[fire_offset] = probability

    for target_position in range(n_pos):
        var channel = WUMPUS_N_FEATURE_CHANNELS + target_position
        for position in range(n_pos):
            var probability = position_false_positive
            if position == target_position:
                probability = position_true_positive
            for sensed in range(2):
                var state = wumpus_state_index(position, sensed, n_pos)
                for theta in range(n_static):
                    var no_fire_offset = (
                        (channel * 2) * n_states + state
                    ) * n_static + theta
                    var fire_offset = (
                        (channel * 2 + 1) * n_states + state
                    ) * n_static + theta
                    result[no_fire_offset] = 1.0 - probability
                    result[fire_offset] = probability
    return result^


def generate_wumpus_goal(
    grid_size: Int,
    pits: List[Float32],
    wumpus: List[Float32],
    gold: List[Float32],
    n_static: Int,
    gold_reward: Float32,
    pit_penalty: Float32,
    wumpus_penalty: Float32,
    temperature: Float32,
) -> List[Float32]:
    """Return per-theta softmax preferences in (state, theta) order."""
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    debug_assert(len(pits) == n_static * n_pos, "pit shape mismatch")
    debug_assert(len(wumpus) == n_static * n_pos, "wumpus shape mismatch")
    debug_assert(len(gold) == n_static * n_pos, "gold shape mismatch")
    debug_assert(temperature > 0.0, "temperature must be positive")
    var result = _zeros(n_states * n_static)
    for theta in range(n_static):
        var scaled_rewards = List[Float32]()
        for state in range(n_states):
            var position = wumpus_state_position(state, n_pos)
            var reward = Float32(0.0)
            if gold[theta * n_pos + position] > 0.5:
                reward += gold_reward
            if pits[theta * n_pos + position] > 0.5:
                reward -= pit_penalty
            if wumpus[theta * n_pos + position] > 0.5:
                reward -= wumpus_penalty
            scaled_rewards.append(reward / temperature)
        var maximum = scaled_rewards[0]
        for state in range(1, n_states):
            maximum = max(maximum, scaled_rewards[state])
        var total = Float32(0.0)
        for state in range(n_states):
            var probability = exp(scaled_rewards[state] - maximum)
            result[state * n_static + theta] = probability
            total += probability
        for state in range(n_states):
            result[state * n_static + theta] /= total
    return result^


def sample_wumpus_observation(
    observation_tensor: List[Float32],
    uniform_draws: List[Float32],
    state: Int,
    theta: Int,
    n_channels: Int,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Emit binary observations from caller-supplied uniform draws."""
    debug_assert(
        len(observation_tensor) == n_channels * 2 * n_states * n_static,
        "observation shape mismatch",
    )
    debug_assert(len(uniform_draws) == n_channels, "draw shape mismatch")
    var result = List[Float32]()
    for channel in range(n_channels):
        var fire_offset = (
            (channel * 2 + 1) * n_states + state
        ) * n_static + theta
        if uniform_draws[channel] < observation_tensor[fire_offset]:
            result.append(1.0)
        else:
            result.append(0.0)
    return result^


def wumpus_is_terminal(
    state: Int,
    grid_size: Int,
    pits: List[Float32],
    wumpus: List[Float32],
    gold: List[Float32],
    theta: Int,
) -> Bool:
    var n_pos = grid_size * grid_size
    var position = wumpus_state_position(state, n_pos)
    return (
        pits[theta * n_pos + position] > 0.5
        or wumpus[theta * n_pos + position] > 0.5
        or gold[theta * n_pos + position] > 0.5
    )


def wumpus_reward(
    state: Int,
    grid_size: Int,
    pits: List[Float32],
    wumpus: List[Float32],
    gold: List[Float32],
    theta: Int,
) -> Float32:
    var n_pos = grid_size * grid_size
    var position = wumpus_state_position(state, n_pos)
    if gold[theta * n_pos + position] > 0.5:
        return 1.0
    if (
        pits[theta * n_pos + position] > 0.5
        or wumpus[theta * n_pos + position] > 0.5
    ):
        return -1.0
    return 0.0


def wumpus_step(
    state: Int,
    action: Int,
    realized_movement_action: Int,
    theta: Int,
    step_count: Int,
    max_steps: Int,
    grid_size: Int,
    pits: List[Float32],
    wumpus: List[Float32],
    gold: List[Float32],
    observation_tensor: List[Float32],
    uniform_draws: List[Float32],
) -> List[Float32]:
    """Pure Wumpus simulator step with caller-owned randomness.

    Returns `[next_state, next_step_count, reward, terminated01,
    truncated01, obs...]`, matching the Frozen and RockSample adapters.
    """
    var n_pos = grid_size * grid_size
    var n_states = 2 * n_pos
    debug_assert(len(pits) % n_pos == 0, "pit shape mismatch")
    var n_static = len(pits) // n_pos
    debug_assert(
        action >= 0 and action < WUMPUS_N_ACTIONS, "action out of range"
    )
    debug_assert(step_count >= 0 and max_steps > 0, "invalid step budget")
    var realized_action = WUMPUS_SENSE
    if action != WUMPUS_SENSE:
        debug_assert(
            realized_movement_action >= 0 and realized_movement_action < 4,
            "realized movement out of range",
        )
        realized_action = realized_movement_action
    var next_state = wumpus_next_state(
        state, realized_action, grid_size, pits, wumpus, theta
    )
    var next_step_count = step_count + 1
    var terminated = wumpus_is_terminal(
        next_state, grid_size, pits, wumpus, gold, theta
    )
    var truncated = next_step_count >= max_steps and not terminated
    var observation = sample_wumpus_observation(
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
    result.append(
        wumpus_reward(next_state, grid_size, pits, wumpus, gold, theta)
    )
    result.append(Float32(1.0) if terminated else Float32(0.0))
    result.append(Float32(1.0) if truncated else Float32(0.0))
    result.extend(observation^)
    return result^
