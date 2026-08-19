from std.collections import List

from aif_mojo.dyn_channel_loopy_bp import (
    _add_messages,
    _compute_pref_to_x,
    _damp_action_channels,
    _dense_dyn_channels_and_pair,
    _dense_dyn_region_beliefs,
    _dense_reduced_dyn_channel,
    _normalize_action_channels,
    _safe_log_values,
    _zeros,
)
from aif_mojo.numerics import (
    LOG_ZERO,
    logsumexp,
    safe_log,
    safe_log_div,
    softmax,
)


def compute_precise_obs_kernels(
    log_observation: List[Float32],
    log_obs_channels: List[Float32],
    log_marginal_obs_channels: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Compute B(y|x,theta) * r(y|x,theta)^2 / r(y|x)."""
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            for obs_idx in range(n_obs_types):
                for state_idx in range(n_states):
                    var marginal_offset = (
                        (time_idx * n_fov + fov_idx) * n_obs_types + obs_idx
                    ) * n_states + state_idx
                    for static_idx in range(n_static):
                        var observation_offset = (
                            (fov_idx * n_obs_types + obs_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        var channel_offset = (
                            marginal_offset * n_static + static_idx
                        )
                        result.append(
                            log_observation[observation_offset]
                            + log_obs_channels[channel_offset]
                            + safe_log_div(
                                log_obs_channels[channel_offset],
                                log_marginal_obs_channels[marginal_offset],
                            )
                        )
    return result^


def _initial_precise_obs_channels(
    log_observation: List[Float32],
    log_prior_theta: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Return tiled r(y|x,theta), then theta-marginal r(y|x)."""
    var conditional_one = List[Float32]()
    for fov_idx in range(n_fov):
        for obs_idx in range(n_obs_types):
            for state_idx in range(n_states):
                for static_idx in range(n_static):
                    var terms = List[Float32]()
                    for candidate in range(n_obs_types):
                        var offset = (
                            (fov_idx * n_obs_types + candidate) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        terms.append(log_observation[offset])
                    var normalizer = logsumexp(terms)
                    var offset = (
                        (fov_idx * n_obs_types + obs_idx) * n_states + state_idx
                    ) * n_static + static_idx
                    conditional_one.append(log_observation[offset] - normalizer)
    var marginal_one = List[Float32]()
    for fov_idx in range(n_fov):
        for obs_idx in range(n_obs_types):
            for state_idx in range(n_states):
                var obs_terms = List[Float32]()
                for static_idx in range(n_static):
                    var offset = (
                        (fov_idx * n_obs_types + obs_idx) * n_states + state_idx
                    ) * n_static + static_idx
                    obs_terms.append(
                        log_observation[offset] + log_prior_theta[static_idx]
                    )
                var value = logsumexp(obs_terms)
                var norm_terms = List[Float32]()
                for candidate in range(n_obs_types):
                    var theta_terms = List[Float32]()
                    for static_idx in range(n_static):
                        var offset = (
                            (fov_idx * n_obs_types + candidate) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        theta_terms.append(
                            log_observation[offset]
                            + log_prior_theta[static_idx]
                        )
                    norm_terms.append(logsumexp(theta_terms))
                marginal_one.append(value - logsumexp(norm_terms))
    var result = List[Float32]()
    for _ in range(horizon + 1):
        for value in conditional_one:
            result.append(value)
    for _ in range(horizon + 1):
        for value in marginal_one:
            result.append(value)
    return result^


def _compute_precise_obs_to_x(
    log_obs_kernels: List[Float32],
    log_prior_theta: List[Float32],
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
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        terms.append(
                            log_obs_kernels[offset]
                            + log_prior_theta[static_idx]
                        )
                value += logsumexp(terms)
            message.append(value)
        var normalizer = logsumexp(message)
        for value in message:
            result.append(value - normalizer)
    return result^


def _forward_pass_precise(
    log_reduced: List[Float32],
    log_q0: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    log_previous: List[Float32],
    damping: Float32,
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
        var damped = List[Float32]()
        for state_idx in range(n_states):
            damped.append(
                (1.0 - damping)
                * log_previous[(time_idx + 1) * n_states + state_idx]
                + damping * (next_message[state_idx] - normalizer)
            )
        var damped_normalizer = logsumexp(damped)
        for state_idx in range(n_states):
            result[(time_idx + 1) * n_states + state_idx] = (
                damped[state_idx] - damped_normalizer
            )
    return result^


def _backward_pass_precise(
    log_reduced: List[Float32],
    log_fwd: List[Float32],
    log_goal: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    log_previous: List[Float32],
    damping: Float32,
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
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
        var damped = List[Float32]()
        for old_idx in range(n_states):
            damped.append(
                (1.0 - damping) * log_previous[time_idx * n_states + old_idx]
                + damping * (message[old_idx] - normalizer)
            )
        var damped_normalizer = logsumexp(damped)
        for old_idx in range(n_states):
            log_bwd[time_idx * n_states + old_idx] = (
                damped[old_idx] - damped_normalizer
            )

    for value in q_u:
        log_bwd.append(value)
    return log_bwd^


def _compute_precise_obs_regions(
    log_obs_kernels: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_prior_theta: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            for obs_idx in range(n_obs_types):
                for state_idx in range(n_states):
                    for static_idx in range(n_static):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        result.append(
                            log_obs_kernels[offset]
                            + log_fwd[time_idx * n_states + state_idx]
                            + log_bwd[time_idx * n_states + state_idx]
                            + log_prior_theta[static_idx]
                        )
    return result^


def _compute_precise_obs_channels(
    log_obs_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var channel_size = (horizon + 1) * n_fov * n_obs_types * n_states * n_static
    var channels = _zeros(channel_size)
    var marginal_size = (horizon + 1) * n_fov * n_obs_types * n_states
    var marginal = _zeros(marginal_size)

    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            for state_idx in range(n_states):
                for static_idx in range(n_static):
                    var values = List[Float32]()
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        values.append(log_obs_regions[offset])
                    var normalizer = logsumexp(values)
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        channels[offset] = values[obs_idx] - normalizer

                var marginal_values = List[Float32]()
                for obs_idx in range(n_obs_types):
                    var theta_terms = List[Float32]()
                    for static_idx in range(n_static):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        theta_terms.append(log_obs_regions[offset])
                    marginal_values.append(logsumexp(theta_terms))
                var marginal_normalizer = logsumexp(marginal_values)
                for obs_idx in range(n_obs_types):
                    var offset = (
                        (time_idx * n_fov + fov_idx) * n_obs_types + obs_idx
                    ) * n_states + state_idx
                    marginal[offset] = (
                        marginal_values[obs_idx] - marginal_normalizer
                    )

    for value in marginal:
        channels.append(value)
    return channels^


def _damp_obs_channels(
    log_old: List[Float32],
    log_new: List[Float32],
    damping: Float32,
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var result = _zeros(len(log_old))
    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            for state_idx in range(n_states):
                for static_idx in range(n_static):
                    var values = List[Float32]()
                    var valid = List[Bool]()
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
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
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        if valid[obs_idx]:
                            result[offset] = values[obs_idx] - normalizer
                        else:
                            result[offset] = LOG_ZERO
    return result^


def _precise_info_seeking_planning(
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
    theta_goal: Bool,
    record_history: Bool = False,
    convergence_mode: Bool = False,
) -> List[Float32]:
    debug_assert(
        len(transition_tensor) == n_states * n_states * n_static * n_actions,
        "transition tensor shape mismatch",
    )
    debug_assert(
        len(observation_tensor) == n_fov * n_obs_types * n_states * n_static,
        "observation tensor shape mismatch",
    )
    debug_assert(len(action_prior) == n_actions, "action prior shape mismatch")
    var log_q0 = _safe_log_values(q_current_state)
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_observation = _safe_log_values(observation_tensor)
    var log_preference = _safe_log_values(goal)
    var log_action_prior = _safe_log_values(action_prior)
    var log_goal = _zeros(n_states)
    if not theta_goal:
        log_goal = _safe_log_values(goal)

    var log_prior_dyn = List[Float32]()
    for _ in range(horizon):
        for value in log_prior_theta:
            log_prior_dyn.append(value)
    var log_prior_obs = List[Float32]()
    for _ in range(horizon + 1):
        for value in log_prior_theta:
            log_prior_obs.append(value)

    var q_u = _zeros(horizon * n_actions)
    var log_fwd_previous = _zeros((horizon + 1) * n_states)
    var log_bwd_previous = _zeros((horizon + 1) * n_states)
    var log_action_channels = List[Float32]()
    var initial_action_channel = -safe_log(Float32(n_actions))
    for _ in range(horizon * n_states * n_actions):
        log_action_channels.append(initial_action_channel)
    var log_obs_channels = List[Float32]()
    var initial_obs_channel = -safe_log(Float32(n_obs_types))
    for _ in range((horizon + 1) * n_fov * n_obs_types * n_states * n_static):
        log_obs_channels.append(initial_obs_channel)
    var log_marginal_obs_channels = List[Float32]()
    for _ in range((horizon + 1) * n_fov * n_obs_types * n_states):
        log_marginal_obs_channels.append(initial_obs_channel)
    if convergence_mode:
        var initial_obs = _initial_precise_obs_channels(
            log_observation,
            log_prior_theta,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var conditional_size = (
            (horizon + 1) * n_fov * n_obs_types * n_states * n_static
        )
        log_obs_channels = List[Float32]()
        log_marginal_obs_channels = List[Float32]()
        for index in range(conditional_size):
            log_obs_channels.append(initial_obs[index])
        for index in range((horizon + 1) * n_fov * n_obs_types * n_states):
            log_marginal_obs_channels.append(
                initial_obs[conditional_size + index]
            )
    var unit_dyn_channels = _zeros(horizon * n_states * n_states * n_actions)
    var diagnostic_history = List[Float32]()

    for _ in range(n_iterations):
        var log_obs_kernels = compute_precise_obs_kernels(
            log_observation,
            log_obs_channels,
            log_marginal_obs_channels,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var reduced = _dense_reduced_dyn_channel(
            log_transition,
            log_prior_dyn,
            unit_dyn_channels,
            log_action_channels,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var log_obs_to_x = _compute_precise_obs_to_x(
            log_obs_kernels,
            log_prior_theta,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var log_local_to_x = List[Float32]()
        if theta_goal:
            var log_pref_to_x = _compute_pref_to_x(
                log_preference,
                log_prior_obs,
                horizon,
                n_states,
                n_static,
            )
            log_local_to_x = _add_messages(log_obs_to_x, log_pref_to_x)
        else:
            for value in log_obs_to_x:
                log_local_to_x.append(value)

        var log_fwd = _forward_pass_precise(
            reduced,
            log_q0,
            log_action_prior,
            log_local_to_x,
            log_fwd_previous,
            damping,
            horizon,
            n_states,
            n_actions,
        )
        var backward_result = _backward_pass_precise(
            reduced,
            log_fwd,
            log_goal,
            log_action_prior,
            log_local_to_x,
            log_bwd_previous,
            damping,
            horizon,
            n_states,
            n_actions,
        )
        var log_bwd = List[Float32]()
        for index in range((horizon + 1) * n_states):
            log_bwd.append(backward_result[index])
        for index in range(horizon * n_actions):
            q_u[index] = backward_result[(horizon + 1) * n_states + index]

        var channel_and_pair = _dense_dyn_channels_and_pair(
            log_transition,
            log_fwd,
            log_bwd,
            log_local_to_x,
            log_prior_dyn,
            log_action_prior,
            unit_dyn_channels,
            log_action_channels,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var channel_size = horizon * n_states * n_states * n_actions
        var log_pair = List[Float32]()
        for index in range(horizon * n_states * n_actions):
            log_pair.append(channel_and_pair[channel_size + index])
        var raw_action_channels = _normalize_action_channels(
            log_pair, horizon, n_states, n_actions
        )
        var diagnostic_dyn_regions = List[Float32]()
        if record_history:
            diagnostic_dyn_regions = _dense_dyn_region_beliefs(
                log_transition,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_prior_dyn,
                log_action_prior,
                unit_dyn_channels,
                log_action_channels,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        log_action_channels = _damp_action_channels(
            log_action_channels,
            raw_action_channels,
            damping,
            horizon,
            n_states,
            n_actions,
        )

        var log_obs_regions = _compute_precise_obs_regions(
            log_obs_kernels,
            log_fwd,
            log_bwd,
            log_prior_theta,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        if record_history:
            for value in reduced:
                diagnostic_history.append(value)
            for value in log_fwd:
                diagnostic_history.append(value)
            for value in log_bwd:
                diagnostic_history.append(value)
            for value in q_u:
                diagnostic_history.append(value)
            for value in diagnostic_dyn_regions:
                diagnostic_history.append(value)
            for value in log_obs_regions:
                diagnostic_history.append(value)
        var raw_obs_result = _compute_precise_obs_channels(
            log_obs_regions,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var obs_channel_size = (
            (horizon + 1) * n_fov * n_obs_types * n_states * n_static
        )
        var raw_obs_channels = List[Float32]()
        for index in range(obs_channel_size):
            raw_obs_channels.append(raw_obs_result[index])
        var raw_marginal_obs_channels = List[Float32]()
        for index in range((horizon + 1) * n_fov * n_obs_types * n_states):
            raw_marginal_obs_channels.append(
                raw_obs_result[obs_channel_size + index]
            )
        log_obs_channels = _damp_obs_channels(
            log_obs_channels,
            raw_obs_channels,
            damping,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        log_marginal_obs_channels = _damp_obs_channels(
            log_marginal_obs_channels,
            raw_marginal_obs_channels,
            damping,
            horizon,
            n_states,
            1,
            n_fov,
            n_obs_types,
        )
        log_fwd_previous = log_fwd^
        log_bwd_previous = log_bwd^

    var result = List[Float32]()
    var total = Float32(0.0)
    for action_idx in range(n_actions):
        total += q_u[action_idx]
    for action_idx in range(n_actions):
        result.append(q_u[action_idx] / (total + 1.0e-10))
    for value in log_action_channels:
        result.append(value)
    for value in log_obs_channels:
        result.append(value)
    if record_history:
        for value in diagnostic_history:
            result.append(value)
    return result^


def precise_info_seeking_planning_dense(
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
    return _precise_info_seeking_planning(
        q_current_state,
        q_static_state,
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
    )


def precise_info_seeking_planning_dense_theta_goal(
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
    return _precise_info_seeking_planning(
        q_current_state,
        q_static_state,
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
        True,
    )
