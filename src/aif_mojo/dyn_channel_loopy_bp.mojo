from std.collections import List

from aif_mojo.numerics import (
    LOG_ZERO,
    logaddexp,
    logsumexp,
    safe_log,
    safe_log_div,
    softmax,
)
from aif_mojo.convergence_control import (
    append_convergence_metadata,
    combined_channel_residual,
    max_channel_residual,
    next_adaptive_damping,
)
from aif_mojo.sparse_messages import (
    sparse_dyn_to_theta_dyn_channel,
    sparse_dyn_channels_and_pair_dyn_channel,
    sparse_reduced_dyn_channel,
)


def _log_static_input(
    values: List[Float32], already_log: Bool
) -> List[Float32]:
    if already_log:
        return values.copy()
    return _safe_log_values(values)


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


def _compute_obs_to_x(
    log_observation: List[Float32],
    log_cavity_obs: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        var message = List[Float32]()
        for state_idx in range(n_states):
            var value = Float32(0.0)
            for fov_idx in range(n_fov):
                var terms = List[Float32]()
                for obs_idx in range(n_obs_types):
                    for static_idx in range(n_static):
                        var offset = (
                            (fov_idx * n_obs_types + obs_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        terms.append(
                            log_observation[offset]
                            + log_cavity_obs[time_idx * n_static + static_idx]
                        )
                value += logsumexp(terms)
            message.append(value)
        var normalizer = logsumexp(message)
        for state_idx in range(n_states):
            result.append(message[state_idx] - normalizer)
    return result^


def _compute_obs_to_theta(
    log_observation: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_extra_to_x: List[Float32],
    use_extra: Bool,
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        for static_idx in range(n_static):
            var value = Float32(0.0)
            for fov_idx in range(n_fov):
                var terms = List[Float32]()
                for obs_idx in range(n_obs_types):
                    for state_idx in range(n_states):
                        var offset = (
                            (fov_idx * n_obs_types + obs_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        var contribution = (
                            log_observation[offset]
                            + log_fwd[time_idx * n_states + state_idx]
                            + log_bwd[time_idx * n_states + state_idx]
                        )
                        if use_extra:
                            contribution += log_extra_to_x[
                                time_idx * n_states + state_idx
                            ]
                        terms.append(contribution)
                value += logsumexp(terms)
            result.append(value)
    return result^


def _compute_theta_cavities_extended(
    log_prior: List[Float32],
    log_dyn_to_theta: List[Float32],
    log_obs_to_theta: List[Float32],
    log_pref_to_theta: List[Float32],
    use_preference: Bool,
    horizon: Int,
    n_static: Int,
) -> List[Float32]:
    """Return dynamics, observation, then optional preference cavities."""
    var total_dyn = _zeros(n_static)
    var total_obs = _zeros(n_static)
    var total_pref = _zeros(n_static)
    for static_idx in range(n_static):
        for time_idx in range(horizon):
            total_dyn[static_idx] += log_dyn_to_theta[
                time_idx * n_static + static_idx
            ]
        for time_idx in range(horizon + 1):
            total_obs[static_idx] += log_obs_to_theta[
                time_idx * n_static + static_idx
            ]
            if use_preference:
                total_pref[static_idx] += log_pref_to_theta[
                    time_idx * n_static + static_idx
                ]

    var result = List[Float32]()
    for time_idx in range(horizon):
        var cavity = List[Float32]()
        for static_idx in range(n_static):
            cavity.append(
                log_prior[static_idx]
                + total_obs[static_idx]
                + total_pref[static_idx]
                + total_dyn[static_idx]
                - log_dyn_to_theta[time_idx * n_static + static_idx]
            )
        var normalizer = logsumexp(cavity)
        for value in cavity:
            result.append(value - normalizer)

    for time_idx in range(horizon + 1):
        var cavity = List[Float32]()
        for static_idx in range(n_static):
            cavity.append(
                log_prior[static_idx]
                + total_dyn[static_idx]
                + total_pref[static_idx]
                + total_obs[static_idx]
                - log_obs_to_theta[time_idx * n_static + static_idx]
            )
        var normalizer = logsumexp(cavity)
        for value in cavity:
            result.append(value - normalizer)

    if use_preference:
        for time_idx in range(horizon + 1):
            var cavity = List[Float32]()
            for static_idx in range(n_static):
                cavity.append(
                    log_prior[static_idx]
                    + total_dyn[static_idx]
                    + total_obs[static_idx]
                    + total_pref[static_idx]
                    - log_pref_to_theta[time_idx * n_static + static_idx]
                )
            var normalizer = logsumexp(cavity)
            for value in cavity:
                result.append(value - normalizer)
    return result^


def _compute_pref_to_x(
    log_preference: List[Float32],
    log_cavity_pref: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        var message = List[Float32]()
        for state_idx in range(n_states):
            var terms = List[Float32]()
            for static_idx in range(n_static):
                terms.append(
                    log_preference[state_idx * n_static + static_idx]
                    + log_cavity_pref[time_idx * n_static + static_idx]
                )
            message.append(logsumexp(terms))
        var normalizer = logsumexp(message)
        for value in message:
            result.append(value - normalizer)
    return result^


def _compute_pref_to_theta(
    log_preference: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_obs_to_x: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        for static_idx in range(n_static):
            var terms = List[Float32]()
            for state_idx in range(n_states):
                terms.append(
                    log_preference[state_idx * n_static + static_idx]
                    + log_fwd[time_idx * n_states + state_idx]
                    + log_bwd[time_idx * n_states + state_idx]
                    + log_obs_to_x[time_idx * n_states + state_idx]
                )
            result.append(logsumexp(terms))
    return result^


def _add_messages(lhs: List[Float32], rhs: List[Float32]) -> List[Float32]:
    debug_assert(len(lhs) == len(rhs), "message shape mismatch")
    var result = List[Float32]()
    for index in range(len(lhs)):
        result.append(lhs[index] + rhs[index])
    return result^


def _dense_reduced_dyn_channel(
    log_transition: List[Float32],
    log_cavity: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense reduction for T * r(action|old) / r(new|old,action)."""
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
                        var channel_offset = (
                            (time_idx * n_states + old_idx) * n_states + new_idx
                        ) * n_actions + action_idx
                        var action_offset = (
                            time_idx * n_states + old_idx
                        ) * n_actions + action_idx
                        terms.append(
                            safe_log_div(
                                log_transition[transition_offset],
                                log_dyn_channels[channel_offset],
                            )
                            + log_action_channel[action_offset]
                            + log_cavity[time_idx * n_static + static_idx]
                        )
                    result.append(logsumexp(terms))
    return result^


def _dense_dyn_to_theta_dyn_channel(
    log_transition: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_action: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon):
        for static_idx in range(n_static):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                for new_idx in range(n_states):
                    for action_idx in range(n_actions):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var channel_offset = (
                            (time_idx * n_states + old_idx) * n_states + new_idx
                        ) * n_actions + action_idx
                        var action_offset = (
                            time_idx * n_states + old_idx
                        ) * n_actions + action_idx
                        terms.append(
                            safe_log_div(
                                log_transition[transition_offset],
                                log_dyn_channels[channel_offset],
                            )
                            + log_action_channel[action_offset]
                            + log_fwd[time_idx * n_states + old_idx]
                            + log_local_to_x[time_idx * n_states + old_idx]
                            + log_bwd[(time_idx + 1) * n_states + new_idx]
                            + log_local_to_x[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_action[action_idx]
                        )
            result.append(logsumexp(terms))
    return result^


def _dense_dyn_channels_and_pair(
    log_transition: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Return dense dynamic channels followed by the pair marginal."""
    var marginal = List[Float32]()
    for _ in range(horizon * n_states * n_states * n_actions):
        marginal.append(LOG_ZERO)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var marginal_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var action_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    for static_idx in range(n_static):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var contribution = (
                            safe_log_div(
                                log_transition[transition_offset],
                                log_dyn_channels[marginal_offset],
                            )
                            + log_action_channel[action_offset]
                            + log_fwd[time_idx * n_states + old_idx]
                            + log_local_to_x[time_idx * n_states + old_idx]
                            + log_bwd[(time_idx + 1) * n_states + new_idx]
                            + log_local_to_x[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_cavity_dyn[time_idx * n_static + static_idx]
                            + log_action[action_idx]
                        )
                        marginal[marginal_offset] = logaddexp(
                            marginal[marginal_offset], contribution
                        )

    var pair = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var terms = List[Float32]()
                for new_idx in range(n_states):
                    var offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    terms.append(marginal[offset])
                pair.append(logsumexp(terms))

    var result = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var marginal_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var pair_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    result.append(marginal[marginal_offset] - pair[pair_offset])
    for value in pair:
        result.append(value)
    return result^


def _dense_dyn_region_beliefs(
    log_transition: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Return unnormalized log regions in (time, old, new, theta, action)."""
    var result = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for static_idx in range(n_static):
                    for action_idx in range(n_actions):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var channel_offset = (
                            (time_idx * n_states + old_idx) * n_states + new_idx
                        ) * n_actions + action_idx
                        var action_offset = (
                            time_idx * n_states + old_idx
                        ) * n_actions + action_idx
                        result.append(
                            safe_log_div(
                                log_transition[transition_offset],
                                log_dyn_channels[channel_offset],
                            )
                            + log_action_channel[action_offset]
                            + log_fwd[time_idx * n_states + old_idx]
                            + log_local_to_x[time_idx * n_states + old_idx]
                            + log_bwd[(time_idx + 1) * n_states + new_idx]
                            + log_local_to_x[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_cavity_dyn[time_idx * n_static + static_idx]
                            + log_action[action_idx]
                        )
    return result^


def _initial_dense_dyn_channels(
    log_transition: List[Float32],
    log_prior_theta: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Theta-marginalized initial r(new|old,action), tiled over time."""
    var one_step = List[Float32]()
    for old_idx in range(n_states):
        for new_idx in range(n_states):
            for action_idx in range(n_actions):
                var terms = List[Float32]()
                for static_idx in range(n_static):
                    var offset = (
                        (new_idx * n_states + old_idx) * n_static + static_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_transition[offset] + log_prior_theta[static_idx]
                    )
                one_step.append(logsumexp(terms))
    for old_idx in range(n_states):
        for action_idx in range(n_actions):
            var terms = List[Float32]()
            for new_idx in range(n_states):
                terms.append(
                    one_step[
                        (old_idx * n_states + new_idx) * n_actions + action_idx
                    ]
                )
            var normalizer = logsumexp(terms)
            for new_idx in range(n_states):
                one_step[
                    (old_idx * n_states + new_idx) * n_actions + action_idx
                ] -= normalizer
    var result = List[Float32]()
    for _ in range(horizon):
        for value in one_step:
            result.append(value)
    return result^


def _reduce_transition(
    use_sparse: Bool,
    transition_indices: List[Int],
    log_transition: List[Float32],
    log_cavity: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    if use_sparse:
        return sparse_reduced_dyn_channel(
            transition_indices,
            log_cavity,
            log_dyn_channels,
            log_action_channel,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
    return _dense_reduced_dyn_channel(
        log_transition,
        log_cavity,
        log_dyn_channels,
        log_action_channel,
        horizon,
        n_states,
        n_actions,
        n_static,
    )


def _compute_channels_and_pair(
    use_sparse: Bool,
    transition_indices: List[Int],
    log_transition: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    if use_sparse:
        return sparse_dyn_channels_and_pair_dyn_channel(
            transition_indices,
            log_fwd,
            log_bwd,
            log_local_to_x,
            log_cavity_dyn,
            log_action,
            log_dyn_channels,
            log_action_channel,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
    return _dense_dyn_channels_and_pair(
        log_transition,
        log_fwd,
        log_bwd,
        log_local_to_x,
        log_cavity_dyn,
        log_action,
        log_dyn_channels,
        log_action_channel,
        horizon,
        n_states,
        n_actions,
        n_static,
    )


def _forward_pass(
    log_reduced: List[Float32],
    log_q0: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[state_idx] = log_q0[state_idx]
    for time_idx in range(horizon):
        var next_message = List[Float32]()
        for new_idx in range(n_states):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced[offset]
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


def _backward_pass(
    log_reduced: List[Float32],
    log_fwd: List[Float32],
    log_goal: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Return backward messages followed by probability action marginals."""
    var log_bwd = _zeros((horizon + 1) * n_states)
    var q_u = _zeros(horizon * n_actions)
    for state_idx in range(n_states):
        log_bwd[horizon * n_states + state_idx] = log_goal[state_idx]

    for reverse_idx in range(horizon):
        var time_idx = horizon - 1 - reverse_idx
        var action_logits = List[Float32]()
        for action_idx in range(n_actions):
            var terms = List[Float32]()
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    var offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced[offset]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                    )
            action_logits.append(
                logsumexp(terms) + log_action_prior[action_idx]
            )
        var action_probabilities = softmax(action_logits)
        for action_idx in range(n_actions):
            q_u[time_idx * n_actions + action_idx] = action_probabilities[
                action_idx
            ]

        var message = List[Float32]()
        for old_idx in range(n_states):
            var terms = List[Float32]()
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced[offset]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action_prior[action_idx]
                    )
            message.append(logsumexp(terms))
        var normalizer = logsumexp(message)
        for old_idx in range(n_states):
            log_bwd[time_idx * n_states + old_idx] = (
                message[old_idx] - normalizer
            )

    for value in q_u:
        log_bwd.append(value)
    return log_bwd^


def _damp_dyn_channels(
    log_old: List[Float32],
    log_new: List[Float32],
    damping: Float32,
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = _zeros(len(log_old))
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var values = List[Float32]()
                var valid = List[Bool]()
                for new_idx in range(n_states):
                    var offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var is_valid = (
                        log_old[offset] > LOG_ZERO / 2.0
                        or log_new[offset] > LOG_ZERO / 2.0
                    )
                    valid.append(is_valid)
                    if is_valid:
                        values.append(
                            (1.0 - damping) * log_old[offset]
                            + damping * log_new[offset]
                        )
                    else:
                        values.append(LOG_ZERO)
                var normalizer = logsumexp(values)
                for new_idx in range(n_states):
                    var offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    if valid[new_idx]:
                        result[offset] = values[new_idx] - normalizer
                    else:
                        result[offset] = LOG_ZERO
    return result^


def _normalize_action_channels(
    log_pair: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            var values = List[Float32]()
            for action_idx in range(n_actions):
                values.append(
                    log_pair[
                        (time_idx * n_states + old_idx) * n_actions + action_idx
                    ]
                )
            var normalizer = logsumexp(values)
            for value in values:
                result.append(value - normalizer)
    return result^


def _damp_action_channels(
    log_old: List[Float32],
    log_new: List[Float32],
    damping: Float32,
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            var values = List[Float32]()
            var valid = List[Bool]()
            for action_idx in range(n_actions):
                var offset = (
                    time_idx * n_states + old_idx
                ) * n_actions + action_idx
                var is_valid = (
                    log_old[offset] > LOG_ZERO / 2.0
                    or log_new[offset] > LOG_ZERO / 2.0
                )
                valid.append(is_valid)
                if is_valid:
                    values.append(
                        (1.0 - damping) * log_old[offset]
                        + damping * log_new[offset]
                    )
                else:
                    values.append(LOG_ZERO)
            var normalizer = logsumexp(values)
            for action_idx in range(n_actions):
                if valid[action_idx]:
                    result.append(values[action_idx] - normalizer)
                else:
                    result.append(LOG_ZERO)
    return result^


def _dyn_channel_loopy_bp_planning(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
    use_sparse: Bool,
    theta_goal: Bool,
    record_history: Bool = False,
    convergence_mode: Bool = False,
    convergence_tolerance: Float32 = -1.0,
    minimum_iterations: Int = 1,
    record_residuals: Bool = False,
    adaptive_damping: Bool = False,
    minimum_damping: Float32 = 0.05,
    maximum_damping: Float32 = 1.0,
    static_inputs_are_log: Bool = False,
) -> List[Float32]:
    """Shared dense/sparse dyn+action-channel planner.

    The result contains the first-step action distribution followed by
    log r(new_state | old_state, action) in (time, old, new, action) order.
    """
    debug_assert(
        len(q_current_state) == n_states, "current state shape mismatch"
    )
    debug_assert(len(q_static_state) == n_static, "static state shape mismatch")
    if theta_goal:
        debug_assert(
            len(goal) == n_states * n_static, "theta goal shape mismatch"
        )
    else:
        debug_assert(len(goal) == n_states, "goal shape mismatch")
    debug_assert(len(action_prior) == n_actions, "action prior shape mismatch")
    if use_sparse:
        debug_assert(
            len(transition_indices) == n_states * n_actions * n_static,
            "transition index shape mismatch",
        )
    else:
        debug_assert(
            len(transition_tensor)
            == n_states * n_states * n_static * n_actions,
            "transition tensor shape mismatch",
        )
    debug_assert(
        len(observation_tensor) == n_fov * n_obs_types * n_states * n_static,
        "observation shape mismatch",
    )

    var log_q0 = _safe_log_values(q_current_state)
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_observation = _log_static_input(
        observation_tensor, static_inputs_are_log
    )
    var log_transition = _log_static_input(
        transition_tensor, static_inputs_are_log
    )
    var log_preference = _log_static_input(goal, static_inputs_are_log)
    var log_goal = _zeros(n_states)
    if not theta_goal:
        log_goal = _log_static_input(goal, static_inputs_are_log)
    var log_action_prior = _log_static_input(
        action_prior, static_inputs_are_log
    )
    var log_dyn_to_theta = _zeros(horizon * n_static)
    var log_obs_to_theta = _zeros((horizon + 1) * n_static)
    var log_pref_to_theta = _zeros((horizon + 1) * n_static)
    var q_u = _zeros(horizon * n_actions)

    var log_dyn_channels = List[Float32]()
    var initial_dyn_channel = -safe_log(Float32(n_states))
    for _ in range(horizon * n_states * n_states * n_actions):
        log_dyn_channels.append(initial_dyn_channel)
    var log_action_channel = List[Float32]()
    var initial_action_channel = -safe_log(Float32(n_actions))
    for _ in range(horizon * n_states * n_actions):
        log_action_channel.append(initial_action_channel)
    if convergence_mode:
        debug_assert(not use_sparse, "convergence mode requires dense T")
        log_dyn_channels = _initial_dense_dyn_channels(
            log_transition,
            log_prior_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
    var diagnostic_history = List[Float32]()
    var residual_history = List[Float32]()
    var effective_damping = damping
    var last_damping = damping
    var previous_residual = Float32(-1.0)
    var final_residual = Float32(-1.0)
    var iterations_used = 0
    var converged = False

    for iteration_idx in range(n_iterations):
        var cavities = _compute_theta_cavities_extended(
            log_prior_theta,
            log_dyn_to_theta,
            log_obs_to_theta,
            log_pref_to_theta,
            theta_goal,
            horizon,
            n_static,
        )
        var log_cavity_dyn = List[Float32]()
        var log_cavity_obs = List[Float32]()
        var log_cavity_pref = List[Float32]()
        for index in range(horizon * n_static):
            log_cavity_dyn.append(cavities[index])
        for index in range((horizon + 1) * n_static):
            log_cavity_obs.append(cavities[horizon * n_static + index])
            if theta_goal:
                log_cavity_pref.append(
                    cavities[(2 * horizon + 1) * n_static + index]
                )

        var reduced = _reduce_transition(
            use_sparse,
            transition_indices,
            log_transition,
            log_cavity_dyn,
            log_dyn_channels,
            log_action_channel,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var log_obs_to_x = _compute_obs_to_x(
            log_observation,
            log_cavity_obs,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var log_pref_to_x = List[Float32]()
        var log_local_to_x = List[Float32]()
        if theta_goal:
            log_pref_to_x = _compute_pref_to_x(
                log_preference,
                log_cavity_pref,
                horizon,
                n_states,
                n_static,
            )
            log_local_to_x = _add_messages(log_obs_to_x, log_pref_to_x)
        else:
            for value in log_obs_to_x:
                log_local_to_x.append(value)
        var log_fwd = _forward_pass(
            reduced,
            log_q0,
            log_action_prior,
            log_local_to_x,
            horizon,
            n_states,
            n_actions,
        )
        var backward_result = _backward_pass(
            reduced,
            log_fwd,
            log_goal,
            log_action_prior,
            log_local_to_x,
            horizon,
            n_states,
            n_actions,
        )
        var log_bwd = List[Float32]()
        for index in range((horizon + 1) * n_states):
            log_bwd.append(backward_result[index])
        for index in range(horizon * n_actions):
            q_u[index] = backward_result[(horizon + 1) * n_states + index]

        if use_sparse:
            log_dyn_to_theta = sparse_dyn_to_theta_dyn_channel(
                transition_indices,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_action_prior,
                log_dyn_channels,
                log_action_channel,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        else:
            log_dyn_to_theta = _dense_dyn_to_theta_dyn_channel(
                log_transition,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_action_prior,
                log_dyn_channels,
                log_action_channel,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        log_obs_to_theta = _compute_obs_to_theta(
            log_observation,
            log_fwd,
            log_bwd,
            log_pref_to_x,
            theta_goal,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        if theta_goal:
            log_pref_to_theta = _compute_pref_to_theta(
                log_preference,
                log_fwd,
                log_bwd,
                log_obs_to_x,
                horizon,
                n_states,
                n_static,
            )
        var channel_result = _compute_channels_and_pair(
            use_sparse,
            transition_indices,
            log_transition,
            log_fwd,
            log_bwd,
            log_local_to_x,
            log_cavity_dyn,
            log_action_prior,
            log_dyn_channels,
            log_action_channel,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        if record_history:
            debug_assert(not use_sparse, "diagnostic history requires dense T")
            var log_dyn_regions = _dense_dyn_region_beliefs(
                log_transition,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_cavity_dyn,
                log_action_prior,
                log_dyn_channels,
                log_action_channel,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
            for value in reduced:
                diagnostic_history.append(value)
            for value in log_fwd:
                diagnostic_history.append(value)
            for value in log_bwd:
                diagnostic_history.append(value)
            for value in q_u:
                diagnostic_history.append(value)
            for value in log_cavity_dyn:
                diagnostic_history.append(value)
            for value in log_dyn_regions:
                diagnostic_history.append(value)
        var channel_size = horizon * n_states * n_states * n_actions
        var pair_size = horizon * n_states * n_actions
        var raw_dyn_channels = List[Float32]()
        var log_pair = List[Float32]()
        for index in range(channel_size):
            raw_dyn_channels.append(channel_result[index])
        for index in range(pair_size):
            log_pair.append(channel_result[channel_size + index])
        var raw_action_channel = _normalize_action_channels(
            log_pair, horizon, n_states, n_actions
        )
        var damped_dyn_channels = _damp_dyn_channels(
            log_dyn_channels,
            raw_dyn_channels,
            effective_damping,
            horizon,
            n_states,
            n_actions,
        )
        var damped_action_channel = log_action_channel.copy()
        if not convergence_mode:
            damped_action_channel = _damp_action_channels(
                log_action_channel,
                raw_action_channel,
                effective_damping,
                horizon,
                n_states,
                n_actions,
            )
            final_residual = combined_channel_residual(
                log_dyn_channels,
                damped_dyn_channels,
                log_action_channel,
                damped_action_channel,
            )
        else:
            final_residual = max_channel_residual(
                log_dyn_channels, damped_dyn_channels
            )
        log_dyn_channels = damped_dyn_channels^
        log_action_channel = damped_action_channel^
        iterations_used = iteration_idx + 1
        last_damping = effective_damping
        if record_residuals:
            residual_history.append(final_residual)
        var should_stop = (
            convergence_tolerance >= 0.0
            and iterations_used >= minimum_iterations
            and final_residual < convergence_tolerance
        )
        if adaptive_damping:
            effective_damping = next_adaptive_damping(
                effective_damping,
                previous_residual,
                final_residual,
                minimum_damping,
                maximum_damping,
            )
        previous_residual = final_residual
        if should_stop:
            converged = True
            break

    var result = List[Float32]()
    var total = Float32(0.0)
    for action_idx in range(n_actions):
        total += q_u[action_idx]
    for action_idx in range(n_actions):
        result.append(q_u[action_idx] / (total + 1.0e-10))
    for value in log_dyn_channels:
        result.append(value)
    if record_history:
        for value in diagnostic_history:
            result.append(value)
    if record_residuals:
        append_convergence_metadata(
            result,
            converged,
            iterations_used,
            final_residual,
            last_damping,
            residual_history,
        )
    return result^


def dyn_channel_loopy_bp_planning_sparse(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Sparse dyn+action-channel planner for a terminal state goal."""
    var no_transition_tensor = List[Float32]()
    return _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        transition_indices,
        no_transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        horizon,
        n_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        True,
        False,
    )


def dyn_channel_loopy_bp_planning_sparse_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Sparse dyn+action-channel planner with per-step (state, theta) goals."""
    var no_transition_tensor = List[Float32]()
    return _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        transition_indices,
        no_transition_tensor,
        observation_tensor,
        goal_by_static,
        action_prior,
        horizon,
        n_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        True,
        True,
    )


def dyn_channel_loopy_bp_planning_dense(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense dyn+action-channel planner for a terminal state goal."""
    var no_transition_indices = List[Int]()
    return _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        no_transition_indices,
        transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        horizon,
        n_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        False,
        False,
    )


def dyn_channel_loopy_bp_planning_dense_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense dyn+action-channel planner with per-step (state, theta) goals."""
    var no_transition_indices = List[Int]()
    return _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        no_transition_indices,
        transition_tensor,
        observation_tensor,
        goal_by_static,
        action_prior,
        horizon,
        n_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        False,
        True,
    )


def dyn_channel_loopy_bp_planning_dense_until_converged(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    maximum_iterations: Int,
    damping: Float32,
    tolerance: Float32,
    minimum_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
    theta_goal: Bool,
    adaptive_damping: Bool = False,
    minimum_damping: Float32 = 0.05,
    maximum_damping: Float32 = 1.0,
) -> List[Float32]:
    """Dense RM-MP with residual early stopping and optional adaptation.

    Returns the normal action/dynamics-channel payload followed by
    `[converged, iterations, final_residual, final_damping, residuals...]`.
    """
    var no_transition_indices = List[Int]()
    return _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        no_transition_indices,
        transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        horizon,
        maximum_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        False,
        theta_goal,
        False,
        False,
        tolerance,
        minimum_iterations,
        True,
        adaptive_damping,
        minimum_damping,
        maximum_damping,
    )


def dyn_channel_loopy_bp_planning_sparse_until_converged(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    maximum_iterations: Int,
    damping: Float32,
    tolerance: Float32,
    minimum_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
    theta_goal: Bool,
    adaptive_damping: Bool = False,
    minimum_damping: Float32 = 0.05,
    maximum_damping: Float32 = 1.0,
) -> List[Float32]:
    """Sparse RM-MP with the same residual contract as the dense API."""
    return _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        transition_indices,
        List[Float32](),
        observation_tensor,
        goal,
        action_prior,
        horizon,
        maximum_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        True,
        theta_goal,
        False,
        False,
        tolerance,
        minimum_iterations,
        True,
        adaptive_damping,
        minimum_damping,
        maximum_damping,
    )


struct PreparedDenseRMMP[horizon: Int, iterations: Int]:
    """Compile-time-specialized RM-MP with static log inputs cached."""

    var log_transition: List[Float32]
    var log_observation: List[Float32]
    var log_goal: List[Float32]
    var log_action_prior: List[Float32]
    var damping: Float32
    var n_states: Int
    var n_actions: Int
    var n_static: Int
    var n_fov: Int
    var n_obs_types: Int
    var theta_goal: Bool

    def __init__(
        out self,
        transition_tensor: List[Float32],
        observation_tensor: List[Float32],
        goal: List[Float32],
        action_prior: List[Float32],
        damping: Float32,
        n_states: Int,
        n_actions: Int,
        n_static: Int,
        n_fov: Int,
        n_obs_types: Int,
        theta_goal: Bool = False,
    ):
        self.log_transition = _safe_log_values(transition_tensor)
        self.log_observation = _safe_log_values(observation_tensor)
        self.log_goal = _safe_log_values(goal)
        self.log_action_prior = _safe_log_values(action_prior)
        self.damping = damping
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_static = n_static
        self.n_fov = n_fov
        self.n_obs_types = n_obs_types
        self.theta_goal = theta_goal

    def plan(
        self,
        q_current_state: List[Float32],
        q_static_state: List[Float32],
    ) -> List[Float32]:
        return _dyn_channel_loopy_bp_planning(
            q_current_state,
            q_static_state,
            List[Int](),
            self.log_transition,
            self.log_observation,
            self.log_goal,
            self.log_action_prior,
            Self.horizon,
            Self.iterations,
            self.damping,
            self.n_states,
            self.n_actions,
            self.n_static,
            self.n_fov,
            self.n_obs_types,
            False,
            self.theta_goal,
            False,
            False,
            -1.0,
            1,
            False,
            False,
            0.05,
            1.0,
            True,
        )


def dyn_channel_loopy_bp_planning_dense_specialized[
    horizon: Int, iterations: Int
](
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
    theta_goal: Bool = False,
) -> List[Float32]:
    """Specialize RM-MP on a compile-time horizon and iteration budget."""
    return _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        List[Int](),
        transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        horizon,
        iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        False,
        theta_goal,
    )
