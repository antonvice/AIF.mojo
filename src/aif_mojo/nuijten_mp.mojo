from std.collections import List
from std.math import exp, log

from aif_mojo.dyn_channel_loopy_bp import (
    _add_messages,
    _compute_pref_to_theta,
    _compute_pref_to_x,
    _compute_theta_cavities_extended,
    _dense_reduced_dyn_channel,
    _safe_log_values,
    _zeros,
)
from aif_mojo.numerics import EPSILON, LOG_ZERO, logsumexp, safe_log, softmax
from aif_mojo.sparse_messages import (
    sparse_dyn_to_theta,
    sparse_efe_action_prior,
    sparse_reduced,
)


def _compute_obs_region_beliefs(
    log_observation: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_cavity_obs: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Normalized probability observation-region beliefs for every timestep."""
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        var log_x = List[Float32]()
        for state_idx in range(n_states):
            log_x.append(
                log_fwd[time_idx * n_states + state_idx]
                + log_bwd[time_idx * n_states + state_idx]
            )
        var x_normalizer = logsumexp(log_x)
        for state_idx in range(n_states):
            log_x[state_idx] -= x_normalizer

        var logits = List[Float32]()
        for fov_idx in range(n_fov):
            for obs_idx in range(n_obs_types):
                for state_idx in range(n_states):
                    for static_idx in range(n_static):
                        var observation_offset = (
                            (fov_idx * n_obs_types + obs_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        logits.append(
                            log_observation[observation_offset]
                            + log_x[state_idx]
                            + log_cavity_obs[time_idx * n_static + static_idx]
                        )
        var probabilities = softmax(logits)
        for probability in probabilities:
            result.append(probability)
    return result^


def _compute_obs_efe_to_x(
    obs_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        var logits = List[Float32]()
        for state_idx in range(n_states):
            var total_entropy = Float32(0.0)
            for fov_idx in range(n_fov):
                var normalizer = Float32(0.0)
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
                        normalizer += obs_regions[offset]
                normalizer += EPSILON
                for static_idx in range(n_static):
                    var theta_marginal = Float32(0.0)
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        theta_marginal += obs_regions[offset] / normalizer
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        var probability = obs_regions[offset] / normalizer
                        var conditional = probability / (
                            theta_marginal + EPSILON
                        )
                        total_entropy -= probability * log(
                            conditional + EPSILON
                        )
            logits.append(-total_entropy)
        var normalizer = logsumexp(logits)
        for value in logits:
            result.append(value - normalizer)
    return result^


def _compute_obs_efe_to_theta(
    obs_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon + 1):
        var logits = List[Float32]()
        for static_idx in range(n_static):
            var total_entropy = Float32(0.0)
            for fov_idx in range(n_fov):
                var normalizer = Float32(0.0)
                for obs_idx in range(n_obs_types):
                    for state_idx in range(n_states):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        normalizer += obs_regions[offset]
                normalizer += EPSILON
                for state_idx in range(n_states):
                    var state_marginal = Float32(0.0)
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        state_marginal += obs_regions[offset] / normalizer
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        var probability = obs_regions[offset] / normalizer
                        var conditional = probability / (
                            state_marginal + EPSILON
                        )
                        total_entropy -= probability * log(
                            conditional + EPSILON
                        )
            logits.append(-total_entropy)
        var probabilities = softmax(logits)
        for probability in probabilities:
            result.append(log(probability + EPSILON))
    return result^


def _forward_pass_nuijten(
    log_reduced: List[Float32],
    log_q0: List[Float32],
    log_action_per_t: List[Float32],
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
                        + log_action_per_t[time_idx * n_actions + action_idx]
                    )
            next_message.append(logsumexp(terms))
        var normalizer = logsumexp(next_message)
        for state_idx in range(n_states):
            result[(time_idx + 1) * n_states + state_idx] = (
                next_message[state_idx] - normalizer
            )
    return result^


def _backward_pass_nuijten(
    log_reduced: List[Float32],
    log_fwd: List[Float32],
    log_goal: List[Float32],
    log_action_per_t: List[Float32],
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
                logsumexp(terms)
                + log_action_per_t[time_idx * n_actions + action_idx]
            )
        var probabilities = softmax(action_logits)
        for action_idx in range(n_actions):
            q_u[time_idx * n_actions + action_idx] = probabilities[action_idx]

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
                        + log_action_per_t[time_idx * n_actions + action_idx]
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


def _compute_dense_dyn_to_theta(
    log_transition: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_action_per_t: List[Float32],
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
                        terms.append(
                            log_transition[transition_offset]
                            + log_fwd[time_idx * n_states + old_idx]
                            + log_local_to_x[time_idx * n_states + old_idx]
                            + log_bwd[(time_idx + 1) * n_states + new_idx]
                            + log_local_to_x[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_action_per_t[
                                time_idx * n_actions + action_idx
                            ]
                        )
            result.append(logsumexp(terms))
    return result^


def _compute_dense_dyn_regions(
    log_transition: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action_per_t: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
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
                        result.append(
                            log_transition[transition_offset]
                            + log_fwd[time_idx * n_states + old_idx]
                            + log_local_to_x[time_idx * n_states + old_idx]
                            + log_bwd[(time_idx + 1) * n_states + new_idx]
                            + log_local_to_x[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_cavity_dyn[time_idx * n_static + static_idx]
                            + log_action_per_t[
                                time_idx * n_actions + action_idx
                            ]
                        )
    return result^


def _compute_dense_efe_action_prior(
    log_dyn_regions: List[Float32],
    action_mask: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for time_idx in range(horizon):
        var efe = List[Float32]()
        for action_idx in range(n_actions):
            var logits = List[Float32]()
            for old_idx in range(n_states):
                for new_idx in range(n_states):
                    for static_idx in range(n_static):
                        var offset = (
                            (
                                (time_idx * n_states + old_idx) * n_states
                                + new_idx
                            )
                            * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        logits.append(log_dyn_regions[offset])
            var normalizer = logsumexp(logits)
            var entropy = Float32(0.0)
            for old_idx in range(n_states):
                for static_idx in range(n_static):
                    var marginal = Float32(0.0)
                    for new_idx in range(n_states):
                        var offset = (
                            (
                                (time_idx * n_states + old_idx) * n_states
                                + new_idx
                            )
                            * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        marginal += exp(log_dyn_regions[offset] - normalizer)
                    for new_idx in range(n_states):
                        var offset = (
                            (
                                (time_idx * n_states + old_idx) * n_states
                                + new_idx
                            )
                            * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var probability = exp(
                            log_dyn_regions[offset] - normalizer
                        )
                        var conditional = probability / (marginal + EPSILON)
                        entropy -= probability * log(conditional + EPSILON)
            if action_mask[action_idx] > 0.0:
                efe.append(entropy)
            else:
                efe.append(LOG_ZERO)
        var probabilities = softmax(efe)
        for probability in probabilities:
            result.append(probability)
    return result^


def _compute_reduced(
    use_sparse: Bool,
    transition_indices: List[Int],
    log_transition: List[Float32],
    log_cavity_dyn: List[Float32],
    unit_dyn_channels: List[Float32],
    unit_action_channels: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    if use_sparse:
        return sparse_reduced(
            transition_indices,
            log_cavity_dyn,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
    return _dense_reduced_dyn_channel(
        log_transition,
        log_cavity_dyn,
        unit_dyn_channels,
        unit_action_channels,
        horizon,
        n_states,
        n_actions,
        n_static,
    )


def _nuijten_mp_planning(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
    use_sparse: Bool,
    theta_goal: Bool,
    record_history: Bool = False,
) -> List[Float32]:
    debug_assert(len(action_prior) == n_actions, "action prior shape mismatch")
    var log_q0 = _safe_log_values(q_current_state)
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_observation = _safe_log_values(observation_tensor)
    var log_preference = _safe_log_values(goal)
    var log_goal = _zeros(n_states)
    if not theta_goal:
        log_goal = _safe_log_values(goal)
    var action_mask = List[Float32]()
    for probability in action_prior:
        if probability > 0.0:
            action_mask.append(1.0)
        else:
            action_mask.append(0.0)

    var log_dyn_to_theta = _zeros(horizon * n_static)
    var log_pref_to_theta = _zeros((horizon + 1) * n_static)
    var q_u = _zeros(horizon * n_actions)
    var action_prior_per_t = List[Float32]()
    for _ in range(horizon):
        for probability in action_prior:
            action_prior_per_t.append(probability)
    var obs_regions = _zeros(
        (horizon + 1) * n_fov * n_obs_types * n_states * n_static
    )
    var log_dyn_regions = _zeros(
        horizon * n_states * n_states * n_static * n_actions
    )
    var unit_dyn_channels = _zeros(horizon * n_states * n_states * n_actions)
    var unit_action_channels = _zeros(horizon * n_states * n_actions)
    var diagnostic_history = List[Float32]()

    for _ in range(n_iterations):
        var action_prior_for_vfe = List[Float32]()
        if record_history:
            for value in action_prior_per_t:
                action_prior_for_vfe.append(value)
        var log_action_per_t = _safe_log_values(action_prior_per_t)
        var log_obs_to_x = _compute_obs_efe_to_x(
            obs_regions,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var log_obs_to_theta = _compute_obs_efe_to_theta(
            obs_regions,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
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
        var reduced = _compute_reduced(
            use_sparse,
            transition_indices,
            log_transition,
            log_cavity_dyn,
            unit_dyn_channels,
            unit_action_channels,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var log_local_to_x = List[Float32]()
        if theta_goal:
            var log_pref_to_x = _compute_pref_to_x(
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

        var log_fwd = _forward_pass_nuijten(
            reduced,
            log_q0,
            log_action_per_t,
            log_local_to_x,
            horizon,
            n_states,
            n_actions,
        )
        var backward_result = _backward_pass_nuijten(
            reduced,
            log_fwd,
            log_goal,
            log_action_per_t,
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
            log_dyn_to_theta = sparse_dyn_to_theta(
                transition_indices,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_action_per_t,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        else:
            log_dyn_to_theta = _compute_dense_dyn_to_theta(
                log_transition,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_action_per_t,
                horizon,
                n_states,
                n_actions,
                n_static,
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
        obs_regions = _compute_obs_region_beliefs(
            log_observation,
            log_fwd,
            log_bwd,
            log_cavity_obs,
            horizon,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        if use_sparse:
            action_prior_per_t = sparse_efe_action_prior(
                transition_indices,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_cavity_dyn,
                log_action_per_t,
                action_mask,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        else:
            log_dyn_regions = _compute_dense_dyn_regions(
                log_transition,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_cavity_dyn,
                log_action_per_t,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
            action_prior_per_t = _compute_dense_efe_action_prior(
                log_dyn_regions,
                action_mask,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        if record_history:
            debug_assert(not use_sparse, "diagnostic history requires dense T")
            for value in log_fwd:
                diagnostic_history.append(value)
            for value in log_bwd:
                diagnostic_history.append(value)
            for value in q_u:
                diagnostic_history.append(value)
            for value in log_dyn_regions:
                diagnostic_history.append(value)
            for value in obs_regions:
                diagnostic_history.append(value)
            for value in log_cavity_dyn:
                diagnostic_history.append(value)
            for value in log_cavity_obs:
                diagnostic_history.append(value)
            for value in action_prior_for_vfe:
                diagnostic_history.append(value)

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    if not use_sparse:
        for value in log_dyn_regions:
            result.append(value)
    for value in obs_regions:
        result.append(value)
    if record_history:
        for value in diagnostic_history:
            result.append(value)
    return result^


def nuijten_mp_planning_sparse(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var no_transition_tensor = List[Float32]()
    return _nuijten_mp_planning(
        q_current_state,
        q_static_state,
        transition_indices,
        no_transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        horizon,
        n_iterations,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        True,
        False,
    )


def nuijten_mp_planning_sparse_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var no_transition_tensor = List[Float32]()
    return _nuijten_mp_planning(
        q_current_state,
        q_static_state,
        transition_indices,
        no_transition_tensor,
        observation_tensor,
        goal_by_static,
        action_prior,
        horizon,
        n_iterations,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        True,
        True,
    )


def nuijten_mp_planning_dense(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var no_transition_indices = List[Int]()
    return _nuijten_mp_planning(
        q_current_state,
        q_static_state,
        no_transition_indices,
        transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        horizon,
        n_iterations,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        False,
        False,
    )


def nuijten_mp_planning_dense_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    var no_transition_indices = List[Int]()
    return _nuijten_mp_planning(
        q_current_state,
        q_static_state,
        no_transition_indices,
        transition_tensor,
        observation_tensor,
        goal_by_static,
        action_prior,
        horizon,
        n_iterations,
        n_states,
        n_actions,
        n_static,
        n_fov,
        n_obs_types,
        False,
        True,
    )
