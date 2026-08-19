from std.collections import List

from aif_mojo.numerics import logsumexp, safe_log, softmax
from aif_mojo.sparse_messages import sparse_dyn_to_theta, sparse_reduced


def _zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def _safe_log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
    return result^


def forward_pass(
    log_reduced_per_t: List[Float32],
    log_q_x0: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Propagate normalized log state messages forward through time."""
    debug_assert(
        len(log_reduced_per_t) == horizon * n_states * n_states * n_actions,
        "reduced transition shape mismatch",
    )
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[state_idx] = log_q_x0[state_idx]

    for time_idx in range(horizon):
        var next_message = List[Float32]()
        for new_idx in range(n_states):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[time_idx * n_states + old_idx]
                        + log_action_prior[action_idx]
                    )
            next_message.append(logsumexp(terms))
        var normalizer = logsumexp(next_message)
        for state_idx in range(n_states):
            result[(time_idx + 1) * n_states + state_idx] = (
                next_message[state_idx] - normalizer
            )
    return result^


def backward_messages(
    log_reduced_per_t: List[Float32],
    log_goal: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Propagate normalized log goal messages backward through time."""
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[horizon * n_states + state_idx] = log_goal[state_idx]

    for reverse_idx in range(horizon):
        var time_idx = horizon - 1 - reverse_idx
        var message = List[Float32]()
        for old_idx in range(n_states):
            var terms = List[Float32]()
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[(time_idx + 1) * n_states + new_idx]
                        + log_action_prior[action_idx]
                    )
            message.append(logsumexp(terms))
        var normalizer = logsumexp(message)
        for state_idx in range(n_states):
            result[time_idx * n_states + state_idx] = (
                message[state_idx] - normalizer
            )
    return result^


def _forward_pass_with_local(
    log_reduced_per_t: List[Float32],
    log_q_x0: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[state_idx] = log_q_x0[state_idx]

    for time_idx in range(horizon):
        var next_message = List[Float32]()
        for new_idx in range(n_states):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_action_prior[action_idx]
                    )
            next_message.append(logsumexp(terms))
        var normalizer = logsumexp(next_message)
        for state_idx in range(n_states):
            result[(time_idx + 1) * n_states + state_idx] = (
                next_message[state_idx] - normalizer
            )
    return result^


def _backward_messages_with_local(
    log_reduced_per_t: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = _zeros((horizon + 1) * n_states)
    for reverse_idx in range(horizon):
        var time_idx = horizon - 1 - reverse_idx
        var message = List[Float32]()
        for old_idx in range(n_states):
            var terms = List[Float32]()
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action_prior[action_idx]
                    )
            message.append(logsumexp(terms))
        var normalizer = logsumexp(message)
        for state_idx in range(n_states):
            result[time_idx * n_states + state_idx] = (
                message[state_idx] - normalizer
            )
    return result^


def _add_messages(lhs: List[Float32], rhs: List[Float32]) -> List[Float32]:
    debug_assert(len(lhs) == len(rhs), "message shape mismatch")
    var result = List[Float32]()
    for i in range(len(lhs)):
        result.append(lhs[i] + rhs[i])
    return result^


def compute_action_marginals(
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Compute probability-space action marginals for every time step."""
    var result = List[Float32]()
    for time_idx in range(horizon):
        var action_logits = List[Float32]()
        for action_idx in range(n_actions):
            var terms = List[Float32]()
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + log_bwd_messages[(time_idx + 1) * n_states + new_idx]
                        + log_fwd_messages[time_idx * n_states + old_idx]
                    )
            action_logits.append(
                logsumexp(terms) + log_action_prior[action_idx]
            )
        var probabilities = softmax(action_logits)
        for action_idx in range(n_actions):
            result.append(probabilities[action_idx])
    return result^


def compute_theta_cavities(
    log_prior_theta: List[Float32],
    log_dyn_to_theta: List[Float32],
    horizon: Int,
    n_static: Int,
) -> List[Float32]:
    """Normalize prior plus every dynamics message except the local factor."""
    debug_assert(
        len(log_dyn_to_theta) == horizon * n_static,
        "dynamics message shape mismatch",
    )
    var totals = List[Float32]()
    for static_idx in range(n_static):
        var value = log_prior_theta[static_idx]
        for time_idx in range(horizon):
            value += log_dyn_to_theta[time_idx * n_static + static_idx]
        totals.append(value)

    var result = List[Float32]()
    for time_idx in range(horizon):
        var cavity = List[Float32]()
        for static_idx in range(n_static):
            cavity.append(
                totals[static_idx]
                - log_dyn_to_theta[time_idx * n_static + static_idx]
            )
        var normalizer = logsumexp(cavity)
        for static_idx in range(n_static):
            result.append(cavity[static_idx] - normalizer)
    return result^


def compute_reduced_per_t_dense(
    log_transition: List[Float32],
    log_cavity_theta: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Marginalize theta from dense (new, old, theta, action) transitions."""
    var result = List[Float32]()
    for time_idx in range(horizon):
        for new_idx in range(n_states):
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var terms = List[Float32]()
                    for static_idx in range(n_static):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        terms.append(
                            log_transition[transition_offset]
                            + log_cavity_theta[time_idx * n_static + static_idx]
                        )
                    result.append(logsumexp(terms))
    return result^


def compute_dyn_to_theta_dense(
    log_transition: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Compute dense dynamics-factor messages to theta."""
    var result = List[Float32]()
    for time_idx in range(horizon):
        for static_idx in range(n_static):
            var terms = List[Float32]()
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    for action_idx in range(n_actions):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        terms.append(
                            log_transition[transition_offset]
                            + log_fwd_messages[time_idx * n_states + old_idx]
                            + log_bwd_messages[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_action_prior[action_idx]
                        )
            result.append(logsumexp(terms))
    return result^


def loopy_bp_planning_sparse(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Sparse loopy-BP planner for the original one-dimensional terminal goal.
    """
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal)
    var log_action_prior = _safe_log_values(action_prior)

    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var local_messages = _zeros((horizon + 1) * n_states)
    var action_per_t = List[Float32]()
    for _ in range(horizon):
        for action_idx in range(n_actions):
            action_per_t.append(log_action_prior[action_idx])

    var q_u = _zeros(horizon * n_actions)
    for _ in range(n_iterations):
        var reduced = sparse_reduced(
            transition_indices,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var log_fwd = forward_pass(
            reduced, log_q0, log_action_prior, horizon, n_states, n_actions
        )
        var log_bwd = backward_messages(
            reduced, log_goal, log_action_prior, horizon, n_states, n_actions
        )
        q_u = compute_action_marginals(
            reduced,
            log_fwd,
            log_bwd,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        var log_dyn_to_theta = sparse_dyn_to_theta(
            transition_indices,
            log_fwd,
            log_bwd,
            local_messages,
            action_per_t,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, log_dyn_to_theta, horizon, n_static
        )

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    return result^


def loopy_bp_planning_dense(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense Loopy BP for the original one-dimensional terminal goal."""
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal)
    var log_action_prior = _safe_log_values(action_prior)
    var log_transition = _safe_log_values(transition_tensor)

    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var q_u = _zeros(horizon * n_actions)
    for _ in range(n_iterations):
        var reduced = compute_reduced_per_t_dense(
            log_transition,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var log_fwd = forward_pass(
            reduced, log_q0, log_action_prior, horizon, n_states, n_actions
        )
        var log_bwd = backward_messages(
            reduced, log_goal, log_action_prior, horizon, n_states, n_actions
        )
        q_u = compute_action_marginals(
            reduced,
            log_fwd,
            log_bwd,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        var log_dyn_to_theta = compute_dyn_to_theta_dense(
            log_transition,
            log_fwd,
            log_bwd,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, log_dyn_to_theta, horizon, n_static
        )

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    return result^


def loopy_bp_planning_dense_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense Loopy BP with a (state, static) preference at every time step."""
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal_by_static)
    var log_action_prior = _safe_log_values(action_prior)
    var log_transition = _safe_log_values(transition_tensor)

    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var previous_dyn_to_theta = _zeros(horizon * n_static)
    var q_u = _zeros(horizon * n_actions)
    for _ in range(n_iterations):
        var reduced = compute_reduced_per_t_dense(
            log_transition,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )

        var theta_logits = List[Float32]()
        for static_idx in range(n_static):
            var value = log_prior_theta[static_idx]
            for time_idx in range(horizon):
                value += previous_dyn_to_theta[time_idx * n_static + static_idx]
            theta_logits.append(value)
        var theta_normalizer = logsumexp(theta_logits)
        for static_idx in range(n_static):
            theta_logits[static_idx] -= theta_normalizer

        var log_preference = List[Float32]()
        for state_idx in range(n_states):
            var terms = List[Float32]()
            for static_idx in range(n_static):
                terms.append(
                    log_goal[state_idx * n_static + static_idx]
                    + theta_logits[static_idx]
                )
            log_preference.append(logsumexp(terms))
        var preference_normalizer = logsumexp(log_preference)
        for state_idx in range(n_states):
            log_preference[state_idx] -= preference_normalizer

        var local_messages = List[Float32]()
        for _ in range(horizon + 1):
            for state_idx in range(n_states):
                local_messages.append(log_preference[state_idx])

        var log_fwd = _forward_pass_with_local(
            reduced,
            log_q0,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var log_bwd = _backward_messages_with_local(
            reduced,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var fwd_with_local = _add_messages(log_fwd, local_messages)
        var bwd_with_local = _add_messages(log_bwd, local_messages)
        q_u = compute_action_marginals(
            reduced,
            fwd_with_local,
            bwd_with_local,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        previous_dyn_to_theta = compute_dyn_to_theta_dense(
            log_transition,
            fwd_with_local,
            bwd_with_local,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, previous_dyn_to_theta, horizon, n_static
        )

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    return result^


def loopy_bp_planning_sparse_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Sparse Loopy BP with a (state, static) preference at every time step."""
    debug_assert(
        len(goal_by_static) == n_states * n_static,
        "theta-dependent goal shape mismatch",
    )
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal_by_static)
    var log_action_prior = _safe_log_values(action_prior)

    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var action_per_t = List[Float32]()
    for _ in range(horizon):
        for action_idx in range(n_actions):
            action_per_t.append(log_action_prior[action_idx])

    var previous_dyn_to_theta = _zeros(horizon * n_static)
    var q_u = _zeros(horizon * n_actions)
    for _ in range(n_iterations):
        var reduced = sparse_reduced(
            transition_indices,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )

        var theta_logits = List[Float32]()
        for static_idx in range(n_static):
            var value = log_prior_theta[static_idx]
            for time_idx in range(horizon):
                value += previous_dyn_to_theta[time_idx * n_static + static_idx]
            theta_logits.append(value)
        var log_q_theta_normalizer = logsumexp(theta_logits)
        for static_idx in range(n_static):
            theta_logits[static_idx] -= log_q_theta_normalizer

        var log_preference = List[Float32]()
        for state_idx in range(n_states):
            var terms = List[Float32]()
            for static_idx in range(n_static):
                terms.append(
                    log_goal[state_idx * n_static + static_idx]
                    + theta_logits[static_idx]
                )
            log_preference.append(logsumexp(terms))
        var preference_normalizer = logsumexp(log_preference)
        for state_idx in range(n_states):
            log_preference[state_idx] -= preference_normalizer

        var local_messages = List[Float32]()
        for _ in range(horizon + 1):
            for state_idx in range(n_states):
                local_messages.append(log_preference[state_idx])

        var log_fwd = _forward_pass_with_local(
            reduced,
            log_q0,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var log_bwd = _backward_messages_with_local(
            reduced,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var fwd_with_local = _add_messages(log_fwd, local_messages)
        var bwd_with_local = _add_messages(log_bwd, local_messages)
        q_u = compute_action_marginals(
            reduced,
            fwd_with_local,
            bwd_with_local,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        previous_dyn_to_theta = sparse_dyn_to_theta(
            transition_indices,
            log_fwd,
            log_bwd,
            local_messages,
            action_per_t,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, previous_dyn_to_theta, horizon, n_static
        )

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    return result^
