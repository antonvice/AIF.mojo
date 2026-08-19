from std.collections import List
from std.math import exp, log


comptime ROCK_LEFT = 0
comptime ROCK_DOWN = 1
comptime ROCK_RIGHT = 2
comptime ROCK_UP = 3
comptime ROCK_EVENT_OTHER = 0
comptime ROCK_NO_INFO = 2
comptime ROCK_N_OBS_TYPES = 3
comptime ROCK_STEP_OBSERVATION_START = 5


def _pow2(exponent: Int) -> Int:
    var result = Int(1)
    for _ in range(exponent):
        result *= 2
    return result


def _zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def rocksample_n_actions(n_rocks: Int) -> Int:
    return n_rocks + 5


def rocksample_sense_action(rock: Int) -> Int:
    return 4 + rock


def rocksample_sample_action(n_rocks: Int) -> Int:
    return 4 + n_rocks


def rocksample_n_events(n_rocks: Int) -> Int:
    return n_rocks + 2


def rocksample_event_sense(rock: Int) -> Int:
    return 1 + rock


def rocksample_event_sample(n_rocks: Int) -> Int:
    return n_rocks + 1


def rocksample_state_index(
    position: Int,
    mask: Int,
    event: Int,
    n_pos: Int,
    n_mask: Int,
    n_events: Int,
) -> Int:
    debug_assert(event < n_events, "event out of range")
    return position + mask * n_pos + event * n_pos * n_mask


def rocksample_state_position(state: Int, n_pos: Int, n_mask: Int) -> Int:
    return (state % (n_pos * n_mask)) % n_pos


def rocksample_state_mask(state: Int, n_pos: Int, n_mask: Int) -> Int:
    return (state % (n_pos * n_mask)) // n_pos


def rocksample_state_event(state: Int, n_pos: Int, n_mask: Int) -> Int:
    return state // (n_pos * n_mask)


def rocksample_chebyshev_distance(
    position_a: Int, position_b: Int, grid_size: Int
) -> Int:
    var row_a = position_a // grid_size
    var column_a = position_a % grid_size
    var row_b = position_b // grid_size
    var column_b = position_b % grid_size
    return max(abs(row_a - row_b), abs(column_a - column_b))


def rocksample_sense_accuracy(
    distance: Int, half_effective_distance: Float32
) -> Float32:
    debug_assert(
        half_effective_distance > 0.0, "half distance must be positive"
    )
    var decay = exp(
        -Float64(distance)
        * log(Float64(2.0))
        / Float64(half_effective_distance)
    )
    return Float32(0.5 + 0.5 * decay)


def rocksample_is_exit(position: Int, grid_size: Int) -> Bool:
    return position % grid_size == grid_size - 1


def generate_rocksample_quality_configs(n_rocks: Int) -> List[Float32]:
    """Return every quality assignment in flat (theta, rock) order."""
    var result = List[Float32]()
    for theta in range(_pow2(n_rocks)):
        for rock in range(n_rocks):
            result.append(Float32((theta >> rock) & 1))
    return result^


def _rock_at(position: Int, rock_positions: List[Int]) -> Int:
    for rock in range(len(rock_positions)):
        if rock_positions[rock] == position:
            return rock
    return -1


def rocksample_next_state(
    state: Int,
    action: Int,
    actual_movement_action: Int,
    grid_size: Int,
    rock_positions: List[Int],
    n_rocks: Int,
) -> Int:
    """Apply an action with an explicitly supplied realized movement action."""
    var n_pos = grid_size * grid_size
    var n_mask = _pow2(n_rocks)
    var n_events = rocksample_n_events(n_rocks)
    var position = rocksample_state_position(state, n_pos, n_mask)
    var mask = rocksample_state_mask(state, n_pos, n_mask)
    if rocksample_is_exit(position, grid_size):
        return state

    var sample = rocksample_sample_action(n_rocks)
    if action >= 4 and action < sample:
        return rocksample_state_index(
            position,
            mask,
            rocksample_event_sense(action - 4),
            n_pos,
            n_mask,
            n_events,
        )
    if action == sample:
        var rock = _rock_at(position, rock_positions)
        if rock >= 0 and (mask & (1 << rock)) == 0:
            return rocksample_state_index(
                position,
                mask | (1 << rock),
                rocksample_event_sample(n_rocks),
                n_pos,
                n_mask,
                n_events,
            )
        return rocksample_state_index(
            position, mask, ROCK_EVENT_OTHER, n_pos, n_mask, n_events
        )

    debug_assert(action >= 0 and action < 4, "action out of range")
    debug_assert(
        actual_movement_action >= 0 and actual_movement_action < 4,
        "realized movement out of range",
    )
    var row = position // grid_size
    var column = position % grid_size
    var new_row = row
    var new_column = column
    if actual_movement_action == ROCK_LEFT:
        new_column -= 1
    elif actual_movement_action == ROCK_DOWN:
        new_row += 1
    elif actual_movement_action == ROCK_RIGHT:
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
    return rocksample_state_index(
        new_position, mask, ROCK_EVENT_OTHER, n_pos, n_mask, n_events
    )


def generate_rocksample_transition(
    grid_size: Int,
    rock_positions: List[Int],
    n_rocks: Int,
    slip_probability: Float32,
) -> List[Float32]:
    """Return dense T(new, old, theta, action)."""
    debug_assert(len(rock_positions) == n_rocks, "rock position shape mismatch")
    var n_pos = grid_size * grid_size
    var n_static = _pow2(n_rocks)
    var n_states = n_pos * n_static * rocksample_n_events(n_rocks)
    var n_actions = rocksample_n_actions(n_rocks)
    var result = _zeros(n_states * n_states * n_static * n_actions)
    for old_state in range(n_states):
        for intended_action in range(n_actions):
            if intended_action < 4:
                for actual_action in range(4):
                    var probability = slip_probability / 3.0
                    if actual_action == intended_action:
                        probability = 1.0 - slip_probability
                    var new_state = rocksample_next_state(
                        old_state,
                        intended_action,
                        actual_action,
                        grid_size,
                        rock_positions,
                        n_rocks,
                    )
                    for theta in range(n_static):
                        var offset = (
                            (new_state * n_states + old_state) * n_static
                            + theta
                        ) * n_actions + intended_action
                        result[offset] += probability
            else:
                var new_state = rocksample_next_state(
                    old_state,
                    intended_action,
                    0,
                    grid_size,
                    rock_positions,
                    n_rocks,
                )
                for theta in range(n_static):
                    var offset = (
                        (new_state * n_states + old_state) * n_static + theta
                    ) * n_actions + intended_action
                    result[offset] = 1.0
    return result^


def generate_rocksample_transition_indices(
    grid_size: Int, rock_positions: List[Int], n_rocks: Int
) -> List[Int]:
    """Return deterministic destinations in flat (old, action, theta) order."""
    var n_static = _pow2(n_rocks)
    var n_states = (
        grid_size * grid_size * n_static * rocksample_n_events(n_rocks)
    )
    var result = List[Int]()
    for old_state in range(n_states):
        for action in range(rocksample_n_actions(n_rocks)):
            var realized = Int(0)
            if action < 4:
                realized = action
            var new_state = rocksample_next_state(
                old_state,
                action,
                realized,
                grid_size,
                rock_positions,
                n_rocks,
            )
            for _ in range(n_static):
                result.append(new_state)
    return result^


def generate_rocksample_observation(
    grid_size: Int,
    rock_positions: List[Int],
    qualities: List[Float32],
    n_rocks: Int,
    half_effective_distance: Float32,
    position_noise: Float32,
) -> List[Float32]:
    """Return B(channel, categorical outcome, state, theta)."""
    var n_pos = grid_size * grid_size
    var n_static = _pow2(n_rocks)
    var n_mask = n_static
    var n_events = rocksample_n_events(n_rocks)
    var n_states = n_pos * n_mask * n_events
    var n_channels = n_pos + n_rocks
    debug_assert(len(rock_positions) == n_rocks, "rock position shape mismatch")
    debug_assert(
        len(qualities) == n_static * n_rocks,
        "quality configuration shape mismatch",
    )
    var result = _zeros(n_channels * ROCK_N_OBS_TYPES * n_states * n_static)

    var true_position = min(max(1.0 - position_noise, 0.01), 0.99)
    var false_position = min(max(position_noise * 0.1, 0.01), 0.99)
    for target_position in range(n_pos):
        for state in range(n_states):
            var position = rocksample_state_position(state, n_pos, n_mask)
            var probability = false_position
            if position == target_position:
                probability = true_position
            for theta in range(n_static):
                var zero_offset = (
                    (target_position * ROCK_N_OBS_TYPES) * n_states + state
                ) * n_static + theta
                var one_offset = (
                    (target_position * ROCK_N_OBS_TYPES + 1) * n_states + state
                ) * n_static + theta
                result[zero_offset] = 1.0 - probability
                result[one_offset] = probability

    for rock in range(n_rocks):
        var channel = n_pos + rock
        var rock_position = rock_positions[rock]
        for state in range(n_states):
            var position = rocksample_state_position(state, n_pos, n_mask)
            var mask = rocksample_state_mask(state, n_pos, n_mask)
            var event = rocksample_state_event(state, n_pos, n_mask)
            for theta in range(n_static):
                if event == rocksample_event_sense(rock):
                    var distance = rocksample_chebyshev_distance(
                        position, rock_position, grid_size
                    )
                    var accuracy = min(
                        max(
                            rocksample_sense_accuracy(
                                distance, half_effective_distance
                            ),
                            0.01,
                        ),
                        0.99,
                    )
                    var quality = Int(qualities[theta * n_rocks + rock])
                    var correct_offset = (
                        (channel * ROCK_N_OBS_TYPES + quality) * n_states
                        + state
                    ) * n_static + theta
                    var incorrect_offset = (
                        (channel * ROCK_N_OBS_TYPES + 1 - quality) * n_states
                        + state
                    ) * n_static + theta
                    result[correct_offset] = accuracy
                    result[incorrect_offset] = 1.0 - accuracy
                elif (
                    event == rocksample_event_sample(n_rocks)
                    and position == rock_position
                    and (mask & (1 << rock)) != 0
                ):
                    var quality = Int(qualities[theta * n_rocks + rock])
                    var correct_offset = (
                        (channel * ROCK_N_OBS_TYPES + quality) * n_states
                        + state
                    ) * n_static + theta
                    var incorrect_offset = (
                        (channel * ROCK_N_OBS_TYPES + 1 - quality) * n_states
                        + state
                    ) * n_static + theta
                    result[correct_offset] = 0.999
                    result[incorrect_offset] = 0.001
                else:
                    var offset = (
                        (channel * ROCK_N_OBS_TYPES + ROCK_NO_INFO) * n_states
                        + state
                    ) * n_static + theta
                    result[offset] = 1.0
    return result^


def generate_rocksample_goal(
    grid_size: Int,
    rock_positions: List[Int],
    qualities: List[Float32],
    n_rocks: Int,
    good_logit: Float32,
    bad_logit: Float32,
    exit_logit: Float32,
    temperature: Float32,
) -> List[Float32]:
    """Return per-config softmax preferences in flat (state, theta) order."""
    debug_assert(temperature > 0.0, "temperature must be positive")
    debug_assert(len(rock_positions) == n_rocks, "rock position shape mismatch")
    var n_pos = grid_size * grid_size
    var n_static = _pow2(n_rocks)
    var n_mask = n_static
    var n_states = n_pos * n_mask * rocksample_n_events(n_rocks)
    debug_assert(
        len(qualities) == n_static * n_rocks,
        "quality configuration shape mismatch",
    )
    var result = _zeros(n_states * n_static)
    for theta in range(n_static):
        var logits = List[Float64]()
        var maximum = Float64(-1.0e300)
        for state in range(n_states):
            var position = rocksample_state_position(state, n_pos, n_mask)
            var mask = rocksample_state_mask(state, n_pos, n_mask)
            var value = Float64(0.0)
            if rocksample_is_exit(position, grid_size):
                value += Float64(exit_logit)
            for rock in range(n_rocks):
                if (mask & (1 << rock)) != 0:
                    if qualities[theta * n_rocks + rock] > 0.5:
                        value += Float64(good_logit)
                    else:
                        value -= Float64(bad_logit)
            logits.append(value / Float64(temperature))
            maximum = max(maximum, logits[state])
        var total = Float64(0.0)
        for state in range(n_states):
            logits[state] = exp(logits[state] - maximum)
            total += logits[state]
        for state in range(n_states):
            result[state * n_static + theta] = Float32(logits[state] / total)
    return result^


def sample_rocksample_observation(
    observation_tensor: List[Float32],
    uniform_draws: List[Float32],
    state: Int,
    theta: Int,
    n_channels: Int,
    n_obs_types: Int,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Emit categorical channel outcomes from B and supplied uniform draws."""
    debug_assert(
        len(observation_tensor)
        == n_channels * n_obs_types * n_states * n_static,
        "observation shape mismatch",
    )
    debug_assert(len(uniform_draws) == n_channels, "draw shape mismatch")
    debug_assert(state >= 0 and state < n_states, "state out of range")
    debug_assert(theta >= 0 and theta < n_static, "theta out of range")
    var result = List[Float32]()
    for channel in range(n_channels):
        var draw = uniform_draws[channel]
        debug_assert(draw >= 0.0 and draw < 1.0, "uniform draw out of range")
        var total = Float32(0.0)
        for outcome in range(n_obs_types):
            var offset = (
                (channel * n_obs_types + outcome) * n_states + state
            ) * n_static + theta
            total += observation_tensor[offset]
        debug_assert(total > 0.0, "observation probabilities sum to zero")
        var threshold = draw * total
        var cumulative = Float32(0.0)
        var selected = n_obs_types - 1
        for outcome in range(n_obs_types):
            var offset = (
                (channel * n_obs_types + outcome) * n_states + state
            ) * n_static + theta
            cumulative += observation_tensor[offset]
            if threshold < cumulative:
                selected = outcome
                break
        result.append(Float32(selected))
    return result^


def rocksample_is_terminal(state: Int, grid_size: Int) -> Bool:
    return rocksample_is_exit(
        rocksample_state_position(state, grid_size * grid_size, 1),
        grid_size,
    )


def rocksample_reward(
    state: Int,
    action: Int,
    next_state: Int,
    theta: Int,
    grid_size: Int,
    rock_positions: List[Int],
    qualities: List[Float32],
    n_rocks: Int,
    good_reward: Float32,
    bad_penalty: Float32,
    exit_reward: Float32,
) -> Float32:
    var n_pos = grid_size * grid_size
    var n_mask = _pow2(n_rocks)
    debug_assert(len(rock_positions) == n_rocks, "rock position shape mismatch")
    debug_assert(
        len(qualities) == n_mask * n_rocks,
        "quality configuration shape mismatch",
    )
    debug_assert(theta >= 0 and theta < n_mask, "theta out of range")
    var reward = Float32(0.0)
    if action == rocksample_sample_action(n_rocks):
        var position = rocksample_state_position(state, n_pos, n_mask)
        var mask = rocksample_state_mask(state, n_pos, n_mask)
        var rock = _rock_at(position, rock_positions)
        if rock >= 0 and (mask & (1 << rock)) == 0:
            if qualities[theta * n_rocks + rock] > 0.5:
                reward += good_reward
            else:
                reward -= bad_penalty
    var next_position = rocksample_state_position(next_state, n_pos, n_mask)
    if rocksample_is_exit(next_position, grid_size):
        reward += exit_reward
    return reward


def rocksample_step(
    state: Int,
    action: Int,
    realized_movement_action: Int,
    theta: Int,
    step_count: Int,
    max_steps: Int,
    grid_size: Int,
    rock_positions: List[Int],
    qualities: List[Float32],
    observation_tensor: List[Float32],
    uniform_draws: List[Float32],
    n_rocks: Int,
    good_reward: Float32,
    bad_penalty: Float32,
    exit_reward: Float32,
) -> List[Float32]:
    """Pure simulator step with explicit movement and observation randomness.

    Returns `[next_state, next_step_count, reward, terminated01,
    truncated01, obs...]`. Supported environment state/count values are exactly
    representable in the flat Float32 adapter; exact state logic remains Int.
    """
    var n_pos = grid_size * grid_size
    var n_static = _pow2(n_rocks)
    var n_events = rocksample_n_events(n_rocks)
    var n_states = n_pos * n_static * n_events
    var n_actions = rocksample_n_actions(n_rocks)
    debug_assert(action >= 0 and action < n_actions, "action out of range")
    debug_assert(theta >= 0 and theta < n_static, "theta out of range")
    debug_assert(step_count >= 0, "step count must be nonnegative")
    debug_assert(max_steps > 0, "max steps must be positive")

    var realized_action = Int(0)
    if action < 4:
        debug_assert(
            realized_movement_action >= 0 and realized_movement_action < 4,
            "realized movement out of range",
        )
        realized_action = realized_movement_action
    var next_state = rocksample_next_state(
        state,
        action,
        realized_action,
        grid_size,
        rock_positions,
        n_rocks,
    )
    var next_step_count = step_count + 1
    var terminated = rocksample_is_terminal(next_state, grid_size)
    var truncated = next_step_count >= max_steps and not terminated
    var observation = sample_rocksample_observation(
        observation_tensor,
        uniform_draws,
        next_state,
        theta,
        n_pos + n_rocks,
        ROCK_N_OBS_TYPES,
        n_states,
        n_static,
    )
    var result = List[Float32]()
    result.append(Float32(next_state))
    result.append(Float32(next_step_count))
    result.append(
        rocksample_reward(
            state,
            action,
            next_state,
            theta,
            grid_size,
            rock_positions,
            qualities,
            n_rocks,
            good_reward,
            bad_penalty,
            exit_reward,
        )
    )
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
