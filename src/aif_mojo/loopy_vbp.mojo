from std.collections import List

from aif_mojo.loopy_bp import (
    _safe_log_values,
    _zeros,
    compute_reduced_per_t_dense,
    compute_theta_cavities,
)
from aif_mojo.numerics import LOG_ZERO, logsumexp, softmax


def _filled(length: Int, value: Float32) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(value)
    return result^


def _backward_pass_vbp_with_local(
    log_reduced_per_t: List[Float32],
    log_terminal: List[Float32],
    log_local: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var log_values = _filled((horizon + 1) * n_states, LOG_ZERO)
    var log_q_values = _filled(horizon * n_states * n_actions, LOG_ZERO)
    for state_idx in range(n_states):
        log_values[horizon * n_states + state_idx] = log_terminal[state_idx]

    for reverse_idx in range(horizon):
        var time_idx = horizon - 1 - reverse_idx
        var next_values = List[Float32]()
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var terms = List[Float32]()
                for new_idx in range(n_states):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + log_values[(time_idx + 1) * n_states + new_idx]
                        + log_local[new_idx]
                    )
                log_q_values[
                    (time_idx * n_states + old_idx) * n_actions + action_idx
                ] = logsumexp(terms)

            var best = log_q_values[(time_idx * n_states + old_idx) * n_actions]
            for action_idx in range(1, n_actions):
                var candidate = log_q_values[
                    (time_idx * n_states + old_idx) * n_actions + action_idx
                ]
                if candidate > best:
                    best = candidate
            next_values.append(best)

        var normalizer = logsumexp(next_values)
        for state_idx in range(n_states):
            log_values[time_idx * n_states + state_idx] = (
                next_values[state_idx] - normalizer
            )

    var result = List[Float32]()
    for value in log_values:
        result.append(value)
    for value in log_q_values:
        result.append(value)
    return result^


def backward_pass_vbp(
    log_reduced_per_t: List[Float32],
    log_goal: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Return normalized log V messages followed by log Q values."""
    debug_assert(
        len(log_reduced_per_t) == horizon * n_states * n_states * n_actions,
        "reduced transition shape mismatch",
    )
    return _backward_pass_vbp_with_local(
        log_reduced_per_t,
        log_goal,
        _zeros(n_states),
        horizon,
        n_states,
        n_actions,
    )


def _forward_pass_vbp_with_local(
    log_reduced_per_t: List[Float32],
    log_q_x0: List[Float32],
    log_q_values: List[Float32],
    log_local: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[state_idx] = log_q_x0[state_idx]

    for time_idx in range(horizon):
        var best_actions = List[Int]()
        for old_idx in range(n_states):
            var best_action = 0
            var best_value = log_q_values[
                (time_idx * n_states + old_idx) * n_actions
            ]
            for action_idx in range(1, n_actions):
                var candidate = log_q_values[
                    (time_idx * n_states + old_idx) * n_actions + action_idx
                ]
                if candidate > best_value:
                    best_action = action_idx
                    best_value = candidate
            best_actions.append(best_action)

        var next_message = List[Float32]()
        for new_idx in range(n_states):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                var action_idx = best_actions[old_idx]
                var reduced_offset = (
                    (time_idx * n_states + new_idx) * n_states + old_idx
                ) * n_actions + action_idx
                terms.append(
                    log_reduced_per_t[reduced_offset]
                    + result[time_idx * n_states + old_idx]
                    + log_local[old_idx]
                )
            next_message.append(logsumexp(terms))
        var normalizer = logsumexp(next_message)
        for state_idx in range(n_states):
            result[(time_idx + 1) * n_states + state_idx] = (
                next_message[state_idx] - normalizer
            )
    return result^


def forward_pass_vbp(
    log_reduced_per_t: List[Float32],
    log_q_x0: List[Float32],
    log_q_values: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Propagate normalized state messages under the greedy Q policy."""
    return _forward_pass_vbp_with_local(
        log_reduced_per_t,
        log_q_x0,
        log_q_values,
        _zeros(n_states),
        horizon,
        n_states,
        n_actions,
    )


def _compute_dyn_to_theta_vbp_with_local(
    log_transition: List[Float32],
    log_fwd_messages: List[Float32],
    log_values: List[Float32],
    log_local: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon):
        for static_idx in range(n_static):
            var state_terms = List[Float32]()
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    var transition_offset = (
                        (new_idx * n_states + old_idx) * n_static + static_idx
                    ) * n_actions
                    var best = (
                        log_transition[transition_offset]
                        + log_fwd_messages[time_idx * n_states + old_idx]
                        + log_local[old_idx]
                        + log_values[(time_idx + 1) * n_states + new_idx]
                        + log_local[new_idx]
                    )
                    for action_idx in range(1, n_actions):
                        var candidate = (
                            log_transition[transition_offset + action_idx]
                            + log_fwd_messages[time_idx * n_states + old_idx]
                            + log_local[old_idx]
                            + log_values[(time_idx + 1) * n_states + new_idx]
                            + log_local[new_idx]
                        )
                        if candidate > best:
                            best = candidate
                    state_terms.append(best)
            result.append(logsumexp(state_terms))
    return result^


def compute_dyn_to_theta_vbp(
    log_transition: List[Float32],
    log_fwd_messages: List[Float32],
    log_values: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Compute dynamics-to-theta messages using max over actions."""
    return _compute_dyn_to_theta_vbp_with_local(
        log_transition,
        log_fwd_messages,
        log_values,
        _zeros(n_states),
        horizon,
        n_states,
        n_actions,
        n_static,
    )


def _split_values(
    backward_result: List[Float32], horizon: Int, n_states: Int
) -> List[Float32]:
    var result = List[Float32]()
    for index in range((horizon + 1) * n_states):
        result.append(backward_result[index])
    return result^


def _split_q_values(
    backward_result: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = List[Float32]()
    var start = (horizon + 1) * n_states
    for index in range(horizon * n_states * n_actions):
        result.append(backward_result[start + index])
    return result^


def _select_first_action(
    log_q0: List[Float32],
    log_values: List[Float32],
    log_q_values: List[Float32],
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var state_weights = List[Float32]()
    for state_idx in range(n_states):
        state_weights.append(log_q0[state_idx] + log_values[state_idx])
    var weights = softmax(state_weights)
    var result = _zeros(n_actions)
    for state_idx in range(n_states):
        var best_action = 0
        var best_value = log_q_values[state_idx * n_actions]
        for action_idx in range(1, n_actions):
            var candidate = log_q_values[state_idx * n_actions + action_idx]
            if candidate > best_value:
                best_action = action_idx
                best_value = candidate
        result[best_action] += weights[state_idx]
    return result^


def loopy_vbp_planning_dense(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    goal: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense Loopy VBP for a one-dimensional terminal goal."""
    var log_q0 = _safe_log_values(q_current_state)
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_goal = _safe_log_values(goal)
    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var log_values = _filled((horizon + 1) * n_states, LOG_ZERO)
    var log_q_values = _filled(horizon * n_states * n_actions, LOG_ZERO)
    for _ in range(n_iterations):
        var reduced = compute_reduced_per_t_dense(
            log_transition,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var backward_result = backward_pass_vbp(
            reduced, log_goal, horizon, n_states, n_actions
        )
        log_values = _split_values(backward_result, horizon, n_states)
        log_q_values = _split_q_values(
            backward_result, horizon, n_states, n_actions
        )
        var log_fwd = forward_pass_vbp(
            reduced, log_q0, log_q_values, horizon, n_states, n_actions
        )
        var log_dyn_to_theta = compute_dyn_to_theta_vbp(
            log_transition,
            log_fwd,
            log_values,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, log_dyn_to_theta, horizon, n_static
        )

    return _select_first_action(
        log_q0, log_values, log_q_values, n_states, n_actions
    )


def loopy_vbp_planning_dense_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    goal_by_static: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense Loopy VBP with a (state, theta) preference at every step."""
    debug_assert(
        len(goal_by_static) == n_states * n_static,
        "theta-dependent goal shape mismatch",
    )
    var log_q0 = _safe_log_values(q_current_state)
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_goal = _safe_log_values(goal_by_static)
    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var previous_dyn_to_theta = _zeros(horizon * n_static)
    var log_values = _filled((horizon + 1) * n_states, LOG_ZERO)
    var log_q_values = _filled(horizon * n_states * n_actions, LOG_ZERO)
    for _ in range(n_iterations):
        var reduced = compute_reduced_per_t_dense(
            log_transition,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )

        var log_theta = List[Float32]()
        for static_idx in range(n_static):
            var value = log_prior_theta[static_idx]
            for time_idx in range(horizon):
                value += previous_dyn_to_theta[time_idx * n_static + static_idx]
            log_theta.append(value)
        var theta_normalizer = logsumexp(log_theta)
        for static_idx in range(n_static):
            log_theta[static_idx] -= theta_normalizer

        var log_preference = List[Float32]()
        for state_idx in range(n_states):
            var terms = List[Float32]()
            for static_idx in range(n_static):
                terms.append(
                    log_goal[state_idx * n_static + static_idx]
                    + log_theta[static_idx]
                )
            log_preference.append(logsumexp(terms))
        var preference_normalizer = logsumexp(log_preference)
        for state_idx in range(n_states):
            log_preference[state_idx] -= preference_normalizer

        var backward_result = _backward_pass_vbp_with_local(
            reduced,
            _zeros(n_states),
            log_preference,
            horizon,
            n_states,
            n_actions,
        )
        log_values = _split_values(backward_result, horizon, n_states)
        log_q_values = _split_q_values(
            backward_result, horizon, n_states, n_actions
        )
        var log_fwd = _forward_pass_vbp_with_local(
            reduced,
            log_q0,
            log_q_values,
            log_preference,
            horizon,
            n_states,
            n_actions,
        )
        previous_dyn_to_theta = _compute_dyn_to_theta_vbp_with_local(
            log_transition,
            log_fwd,
            log_values,
            log_preference,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, previous_dyn_to_theta, horizon, n_static
        )

    return _select_first_action(
        log_q0, log_values, log_q_values, n_states, n_actions
    )
