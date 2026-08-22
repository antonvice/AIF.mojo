from std.collections import List

from aif_mojo.numerics import logsumexp, safe_log


def state_preference_messages(
    preferences: List[Float32],
    horizon: Int,
    n_states: Int,
    terminal_only: Bool = False,
) -> List[Float32]:
    """Return log state preferences in flat `(time, state)` order.

    `preferences` may be one shared `(state)` vector or an explicit
    `(horizon + 1, state)` schedule. A shared terminal preference is applied
    only at `time == horizon`; other time slices are neutral log factors.
    """
    debug_assert(
        len(preferences) == n_states
        or len(preferences) == (horizon + 1) * n_states,
        "state preference shape mismatch",
    )
    debug_assert(
        not terminal_only or len(preferences) == n_states,
        "terminal-only preference must be a shared state vector",
    )
    var result = List[Float32](capacity=(horizon + 1) * n_states)
    for time_idx in range(horizon + 1):
        for state_idx in range(n_states):
            if terminal_only and time_idx < horizon:
                result.append(0.0)
            elif len(preferences) == n_states:
                result.append(safe_log(preferences[state_idx]))
            else:
                result.append(
                    safe_log(preferences[time_idx * n_states + state_idx])
                )
    return result^


def observation_preference_messages(
    observation_tensor: List[Float32],
    q_static_state: List[Float32],
    preferences: List[Float32],
    horizon: Int,
    n_fov: Int,
    n_obs_types: Int,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Map shared or time-indexed observation preferences to state factors.

    The approximation integrates each observation modality against the fixed
    current `q(theta)` and sums modality log messages. B is flattened as
    `(field, outcome, state, theta)`; preferences are `(field, outcome)` or
    `(time, field, outcome)`.
    """
    debug_assert(
        len(observation_tensor) == n_fov * n_obs_types * n_states * n_static,
        "observation tensor shape mismatch",
    )
    debug_assert(len(q_static_state) == n_static, "static belief mismatch")
    var preference_stride = n_fov * n_obs_types
    debug_assert(
        len(preferences) == preference_stride
        or len(preferences) == (horizon + 1) * preference_stride,
        "observation preference shape mismatch",
    )
    var result = List[Float32](capacity=(horizon + 1) * n_states)
    for time_idx in range(horizon + 1):
        var preference_time = 0
        if len(preferences) != preference_stride:
            preference_time = time_idx * preference_stride
        for state_idx in range(n_states):
            var local = Float32(0.0)
            for field_idx in range(n_fov):
                var terms = List[Float32](capacity=n_obs_types * n_static)
                for outcome_idx in range(n_obs_types):
                    var preference_offset = (
                        preference_time + field_idx * n_obs_types + outcome_idx
                    )
                    for static_idx in range(n_static):
                        var observation_offset = (
                            (field_idx * n_obs_types + outcome_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        terms.append(
                            safe_log(observation_tensor[observation_offset])
                            + safe_log(q_static_state[static_idx])
                            + safe_log(preferences[preference_offset])
                        )
                local += logsumexp(terms)
            result.append(local)
    return result^


def combine_preference_messages(
    state_messages: List[Float32],
    observation_messages: List[Float32],
) -> List[Float32]:
    """Add compatible log preference factors; an empty side is neutral."""
    if len(state_messages) == 0:
        return observation_messages.copy()
    if len(observation_messages) == 0:
        return state_messages.copy()
    debug_assert(
        len(state_messages) == len(observation_messages),
        "preference message shape mismatch",
    )
    var result = List[Float32](capacity=len(state_messages))
    for index in range(len(state_messages)):
        result.append(state_messages[index] + observation_messages[index])
    return result^


def preference_time_slice(
    messages: List[Float32], time_idx: Int, n_states: Int
) -> List[Float32]:
    """Extract one state-preference time slice."""
    debug_assert(time_idx >= 0, "negative preference time")
    debug_assert(
        (time_idx + 1) * n_states <= len(messages),
        "preference time out of range",
    )
    var result = List[Float32](capacity=n_states)
    for state_idx in range(n_states):
        result.append(messages[time_idx * n_states + state_idx])
    return result^
