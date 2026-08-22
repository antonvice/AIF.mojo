from std.collections import List

from aif_mojo.dyn_channel_loopy_bp import (
    _add_messages,
    _compute_pref_to_x,
    _damp_action_channels,
    _damp_dyn_channels,
    _dense_dyn_region_beliefs,
    _initial_dense_dyn_channels,
    _normalize_action_channels,
    _safe_log_values,
    _zeros,
)
from aif_mojo.numerics import LOG_ZERO, logsumexp, safe_log, safe_log_div
from aif_mojo.convergence_control import (
    append_convergence_metadata,
    max_channel_residual,
    next_adaptive_damping,
)
from aif_mojo.precise_info_seeking import (
    _backward_pass_precise,
    _compute_precise_obs_regions,
    _compute_precise_obs_to_x,
    _damp_obs_channels,
    _forward_pass_precise,
    _initial_precise_obs_channels,
    compute_precise_obs_kernels,
)


def _log_static_input(
    values: List[Float32], already_log: Bool
) -> List[Float32]:
    if already_log:
        return values.copy()
    return _safe_log_values(values)


def compute_dyn_kernels_aif(
    log_transition_kernel: List[Float32],
    log_action_channels: List[Float32],
    log_dyn_channels: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> List[Float32]:
    """Compute T * r(action|old) / r(new|old,action) in log space.

    The input transition kernel is ordered (old, new, theta, action). The
    result is ordered (time, old, new, theta, action), matching JAX.
    """
    debug_assert(
        len(log_transition_kernel)
        == n_states * n_states * n_static * n_actions,
        "transition kernel shape mismatch",
    )
    debug_assert(
        len(log_action_channels) == horizon * n_states * n_actions,
        "action channel shape mismatch",
    )
    debug_assert(
        len(log_dyn_channels) == horizon * n_states * n_states * n_actions,
        "dynamics channel shape mismatch",
    )
    var result = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for static_idx in range(n_static):
                    for action_idx in range(n_actions):
                        var transition_offset = (
                            (old_idx * n_states + new_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var action_offset = (
                            time_idx * n_states + old_idx
                        ) * n_actions + action_idx
                        var dyn_offset = (
                            (time_idx * n_states + old_idx) * n_states + new_idx
                        ) * n_actions + action_idx
                        var quotient = safe_log_div(
                            log_transition_kernel[transition_offset],
                            log_dyn_channels[dyn_offset],
                        )
                        if quotient > LOG_ZERO / 2.0:
                            result.append(
                                quotient + log_action_channels[action_offset]
                            )
                        else:
                            result.append(LOG_ZERO)
    return result^


def _compute_dense_log_base(
    log_transition: List[Float32],
    log_prior_theta: List[Float32],
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Marginalize theta into (old state, new state, action) order."""
    var result = List[Float32]()
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
                result.append(logsumexp(terms))
    return result^


def _broadcast_local_message(
    log_local_to_x: List[Float32], horizon: Int, n_states: Int
) -> List[Float32]:
    debug_assert(
        len(log_local_to_x) == n_states
        or len(log_local_to_x) == (horizon + 1) * n_states,
        "local state message shape mismatch",
    )
    if len(log_local_to_x) != n_states:
        return log_local_to_x.copy()
    var result = List[Float32]()
    for _ in range(horizon + 1):
        for value in log_local_to_x:
            result.append(value)
    return result^


def _reduced_from_base(
    log_base: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channels: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Apply r(action|old) / r(new|old,action), returning (t,new,old,a)."""
    var result = List[Float32]()
    for time_idx in range(horizon):
        for new_idx in range(n_states):
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var channel_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var action_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    var base_offset = (
                        old_idx * n_states + new_idx
                    ) * n_actions + action_idx
                    if log_dyn_channels[channel_offset] > LOG_ZERO / 2.0:
                        result.append(
                            log_base[base_offset]
                            - log_dyn_channels[channel_offset]
                            + log_action_channels[action_offset]
                        )
                    else:
                        result.append(LOG_ZERO)
    return result^


def _raw_dyn_channels_and_pair(
    log_reduced: List[Float32],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Return raw r(new|old,action), then the (t,old,action) log pair."""
    var joint = _zeros(horizon * n_states * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    var joint_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    joint[joint_offset] = (
                        log_reduced[reduced_offset]
                        + log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action_prior[action_idx]
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
                    terms.append(joint[offset])
                pair.append(logsumexp(terms))

    var result = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var joint_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var pair_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    result.append(joint[joint_offset] - pair[pair_offset])
    for value in pair:
        result.append(value)
    return result^


def _raw_obs_channels(
    log_obs_kernels: List[Float32],
    log_prior_theta: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Return raw r(obs|state,theta), then theta-marginal r(obs|state)."""
    var conditional = _zeros(
        (horizon + 1) * n_fov * n_obs_types * n_states * n_static
    )
    var marginal = _zeros((horizon + 1) * n_fov * n_obs_types * n_states)
    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            for state_idx in range(n_states):
                for static_idx in range(n_static):
                    var terms = List[Float32]()
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        terms.append(log_obs_kernels[offset])
                    var normalizer = logsumexp(terms)
                    for obs_idx in range(n_obs_types):
                        var offset = (
                            (
                                (time_idx * n_fov + fov_idx) * n_obs_types
                                + obs_idx
                            )
                            * n_states
                            + state_idx
                        ) * n_static + static_idx
                        conditional[offset] = terms[obs_idx] - normalizer

                var marginal_terms = List[Float32]()
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
                        theta_terms.append(
                            log_obs_kernels[offset]
                            + log_prior_theta[static_idx]
                        )
                    marginal_terms.append(logsumexp(theta_terms))
                var marginal_normalizer = logsumexp(marginal_terms)
                for obs_idx in range(n_obs_types):
                    var offset = (
                        (time_idx * n_fov + fov_idx) * n_obs_types + obs_idx
                    ) * n_states + state_idx
                    marginal[offset] = (
                        marginal_terms[obs_idx] - marginal_normalizer
                    )

    var result = List[Float32]()
    for value in conditional:
        result.append(value)
    for value in marginal:
        result.append(value)
    return result^


def precompute_obs_channels(
    log_observation: List[Float32],
    log_prior_theta: List[Float32],
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Converge time-invariant observation channels and return log obs->x."""
    debug_assert(
        len(log_observation) == n_fov * n_obs_types * n_states * n_static,
        "observation shape mismatch",
    )
    debug_assert(len(log_prior_theta) == n_static, "static prior mismatch")
    var initial = -safe_log(Float32(n_obs_types))
    var log_obs_channels = List[Float32]()
    for _ in range(n_fov * n_obs_types * n_states * n_static):
        log_obs_channels.append(initial)
    var log_marginal_obs_channels = List[Float32]()
    for _ in range(n_fov * n_obs_types * n_states):
        log_marginal_obs_channels.append(initial)

    for _ in range(n_iterations):
        var kernels = compute_precise_obs_kernels(
            log_observation,
            log_obs_channels,
            log_marginal_obs_channels,
            0,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var raw = _raw_obs_channels(
            kernels,
            log_prior_theta,
            0,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        var conditional_size = n_fov * n_obs_types * n_states * n_static
        var raw_conditional = List[Float32]()
        for index in range(conditional_size):
            raw_conditional.append(raw[index])
        var raw_marginal = List[Float32]()
        for index in range(n_fov * n_obs_types * n_states):
            raw_marginal.append(raw[conditional_size + index])
        log_obs_channels = _damp_obs_channels(
            log_obs_channels,
            raw_conditional,
            damping,
            0,
            n_states,
            n_static,
            n_fov,
            n_obs_types,
        )
        log_marginal_obs_channels = _damp_obs_channels(
            log_marginal_obs_channels,
            raw_marginal,
            damping,
            0,
            n_states,
            1,
            n_fov,
            n_obs_types,
        )

    var converged_kernels = compute_precise_obs_kernels(
        log_observation,
        log_obs_channels,
        log_marginal_obs_channels,
        0,
        n_states,
        n_static,
        n_fov,
        n_obs_types,
    )
    return _compute_precise_obs_to_x(
        converged_kernels,
        log_prior_theta,
        0,
        n_states,
        n_static,
        n_fov,
        n_obs_types,
    )


def precompute_pref_to_x(
    log_preference: List[Float32],
    log_prior_theta: List[Float32],
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Compute the constant normalized preference-to-state message."""
    debug_assert(
        len(log_preference) == n_states * n_static,
        "preference shape mismatch",
    )
    debug_assert(len(log_prior_theta) == n_static, "static prior mismatch")
    return _compute_pref_to_x(
        log_preference, log_prior_theta, 0, n_states, n_static
    )


def _active_inference_planning(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    log_base: List[Float32],
    log_precomputed_local: List[Float32],
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
    log_transition_diagnostics: List[Float32],
    dense_observations: Bool,
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
    """Shared dense and precomputed planner.

    The flattened result is action probabilities, log dynamics channels in
    (time, old, new, action) order, then dense log observation channels in
    (time, field, observation, state, theta) order when requested.
    """
    debug_assert(
        len(q_current_state) == n_states, "current state shape mismatch"
    )
    debug_assert(len(q_static_state) == n_static, "static state mismatch")
    debug_assert(
        len(log_base) == n_states * n_states * n_actions,
        "transition base shape mismatch",
    )
    debug_assert(len(action_prior) == n_actions, "action prior mismatch")
    if theta_goal:
        debug_assert(
            len(goal) == n_states * n_static, "theta goal shape mismatch"
        )
    else:
        debug_assert(len(goal) == n_states, "terminal goal shape mismatch")
    if dense_observations:
        debug_assert(
            len(observation_tensor)
            == n_fov * n_obs_types * n_states * n_static,
            "observation shape mismatch",
        )

    var log_q0 = _safe_log_values(q_current_state)
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_action_prior = _log_static_input(
        action_prior, static_inputs_are_log
    )
    var log_goal = _zeros(n_states)
    if not theta_goal:
        log_goal = _log_static_input(goal, static_inputs_are_log)
    var log_preference = List[Float32]()
    if theta_goal:
        log_preference = _log_static_input(goal, static_inputs_are_log)
    var log_observation = List[Float32]()
    if dense_observations:
        log_observation = _log_static_input(
            observation_tensor, static_inputs_are_log
        )

    var fixed_local = List[Float32]()
    if not dense_observations:
        fixed_local = _broadcast_local_message(
            log_precomputed_local, horizon, n_states
        )
    var log_prior_obs = List[Float32]()
    for _ in range(horizon + 1):
        for value in log_prior_theta:
            log_prior_obs.append(value)
    var log_prior_dyn = List[Float32]()
    for _ in range(horizon):
        for value in log_prior_theta:
            log_prior_dyn.append(value)

    var q_u = _zeros(horizon * n_actions)
    var log_fwd_previous = _zeros((horizon + 1) * n_states)
    var log_bwd_previous = _zeros((horizon + 1) * n_states)
    var initial_action = -safe_log(Float32(n_actions))
    var log_action_channels = List[Float32]()
    for _ in range(horizon * n_states * n_actions):
        log_action_channels.append(initial_action)
    var initial_dyn = -safe_log(Float32(n_states))
    var log_dyn_channels = List[Float32]()
    for _ in range(horizon * n_states * n_states * n_actions):
        log_dyn_channels.append(initial_dyn)

    var log_obs_channels = List[Float32]()
    var log_marginal_obs_channels = List[Float32]()
    if dense_observations:
        var initial_obs = -safe_log(Float32(n_obs_types))
        for _ in range(
            (horizon + 1) * n_fov * n_obs_types * n_states * n_static
        ):
            log_obs_channels.append(initial_obs)
        for _ in range((horizon + 1) * n_fov * n_obs_types * n_states):
            log_marginal_obs_channels.append(initial_obs)
    if convergence_mode:
        log_dyn_channels = _initial_dense_dyn_channels(
            log_transition_diagnostics,
            log_prior_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
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
    var diagnostic_history = List[Float32]()
    var residual_history = List[Float32]()
    var effective_damping = damping
    var last_damping = damping
    var previous_residual = Float32(-1.0)
    var final_residual = Float32(-1.0)
    var iterations_used = 0
    var converged = False

    for iteration_idx in range(n_iterations):
        var log_local_to_x: List[Float32]
        var log_obs_kernels = List[Float32]()
        if dense_observations:
            log_obs_kernels = compute_precise_obs_kernels(
                log_observation,
                log_obs_channels,
                log_marginal_obs_channels,
                horizon,
                n_states,
                n_static,
                n_fov,
                n_obs_types,
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
                log_local_to_x = log_obs_to_x^
        else:
            log_local_to_x = fixed_local.copy()

        var reduced = _reduced_from_base(
            log_base,
            log_dyn_channels,
            log_action_channels,
            horizon,
            n_states,
            n_actions,
        )
        var log_fwd = _forward_pass_precise(
            reduced,
            log_q0,
            log_action_prior,
            log_local_to_x,
            log_fwd_previous,
            effective_damping,
            horizon,
            n_states,
            n_actions,
        )
        var backward = _backward_pass_precise(
            reduced,
            log_fwd,
            log_goal,
            log_action_prior,
            log_local_to_x,
            log_bwd_previous,
            effective_damping,
            horizon,
            n_states,
            n_actions,
        )
        var log_bwd = List[Float32]()
        for index in range((horizon + 1) * n_states):
            log_bwd.append(backward[index])
        for index in range(horizon * n_actions):
            q_u[index] = backward[(horizon + 1) * n_states + index]

        if record_history:
            debug_assert(
                dense_observations, "diagnostics require dense observations"
            )
            debug_assert(
                len(log_transition_diagnostics)
                == n_states * n_states * n_static * n_actions,
                "diagnostic transition shape mismatch",
            )
            var log_dyn_regions = _dense_dyn_region_beliefs(
                log_transition_diagnostics,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_prior_dyn,
                log_action_prior,
                log_dyn_channels,
                log_action_channels,
                horizon,
                n_states,
                n_actions,
                n_static,
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
            for value in reduced:
                diagnostic_history.append(value)
            for value in log_fwd:
                diagnostic_history.append(value)
            for value in log_bwd:
                diagnostic_history.append(value)
            for value in q_u:
                diagnostic_history.append(value)
            for value in log_dyn_regions:
                diagnostic_history.append(value)
            for value in log_obs_regions:
                diagnostic_history.append(value)

        var raw_dyn_and_pair = _raw_dyn_channels_and_pair(
            reduced,
            log_fwd,
            log_bwd,
            log_local_to_x,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        var dyn_size = horizon * n_states * n_states * n_actions
        var raw_dyn = List[Float32]()
        for index in range(dyn_size):
            raw_dyn.append(raw_dyn_and_pair[index])
        var pair = List[Float32]()
        for index in range(horizon * n_states * n_actions):
            pair.append(raw_dyn_and_pair[dyn_size + index])
        var raw_action = _normalize_action_channels(
            pair, horizon, n_states, n_actions
        )
        var damped_action_channels = _damp_action_channels(
            log_action_channels,
            raw_action,
            effective_damping,
            horizon,
            n_states,
            n_actions,
        )
        var damped_dyn_channels = _damp_dyn_channels(
            log_dyn_channels,
            raw_dyn,
            effective_damping,
            horizon,
            n_states,
            n_actions,
        )
        final_residual = max(
            max_channel_residual(log_action_channels, damped_action_channels),
            max_channel_residual(log_dyn_channels, damped_dyn_channels),
        )

        if dense_observations:
            var raw_obs = _raw_obs_channels(
                log_obs_kernels,
                log_prior_theta,
                horizon,
                n_states,
                n_static,
                n_fov,
                n_obs_types,
            )
            var obs_size = (
                (horizon + 1) * n_fov * n_obs_types * n_states * n_static
            )
            var raw_obs_channels = List[Float32]()
            for index in range(obs_size):
                raw_obs_channels.append(raw_obs[index])
            var raw_marginal = List[Float32]()
            for index in range((horizon + 1) * n_fov * n_obs_types * n_states):
                raw_marginal.append(raw_obs[obs_size + index])
            var damped_obs_channels = _damp_obs_channels(
                log_obs_channels,
                raw_obs_channels,
                effective_damping,
                horizon,
                n_states,
                n_static,
                n_fov,
                n_obs_types,
            )
            var damped_marginal_obs_channels = _damp_obs_channels(
                log_marginal_obs_channels,
                raw_marginal,
                effective_damping,
                horizon,
                n_states,
                1,
                n_fov,
                n_obs_types,
            )
            final_residual = max(
                final_residual,
                max(
                    max_channel_residual(log_obs_channels, damped_obs_channels),
                    max_channel_residual(
                        log_marginal_obs_channels,
                        damped_marginal_obs_channels,
                    ),
                ),
            )
            log_obs_channels = damped_obs_channels^
            log_marginal_obs_channels = damped_marginal_obs_channels^
        log_action_channels = damped_action_channels^
        log_dyn_channels = damped_dyn_channels^
        log_fwd_previous = log_fwd^
        log_bwd_previous = log_bwd^
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
    if dense_observations:
        for value in log_obs_channels:
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


def active_inference_planning_precomputed(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    log_base: List[Float32],
    log_local_to_x: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
        log_local_to_x,
        List[Float32](),
        goal,
        action_prior,
        horizon,
        n_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        0,
        0,
        List[Float32](),
        False,
        False,
    )


def active_inference_planning_precomputed_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    log_base: List[Float32],
    log_local_to_x: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
        log_local_to_x,
        List[Float32](),
        goal_by_static,
        action_prior,
        horizon,
        n_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        0,
        0,
        List[Float32](),
        False,
        True,
    )


def active_inference_planning_precomputed_until_converged(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    log_base: List[Float32],
    log_local_to_x: List[Float32],
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
    theta_goal: Bool,
    adaptive_damping: Bool = False,
    minimum_damping: Float32 = 0.05,
    maximum_damping: Float32 = 1.0,
) -> List[Float32]:
    """Precomputed AIF-MP with residual early stopping."""
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
        log_local_to_x,
        List[Float32](),
        goal,
        action_prior,
        horizon,
        maximum_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        0,
        0,
        List[Float32](),
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


def active_inference_planning_precomputed_with_preferences(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    log_base: List[Float32],
    log_observation_local: List[Float32],
    log_preference_local: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Precomputed AIF-MP with explicit time-indexed log preferences.

    Both local inputs may be `(state)` or `(horizon + 1, state)`. The state
    and/or observation preference APIs in `aif_mojo.preferences` produce the
    second input. A neutral terminal goal avoids applying preferences twice.
    """
    var observation_local = _broadcast_local_message(
        log_observation_local, horizon, n_states
    )
    var preference_local = _broadcast_local_message(
        log_preference_local, horizon, n_states
    )
    var combined = List[Float32](capacity=(horizon + 1) * n_states)
    for index in range((horizon + 1) * n_states):
        combined.append(observation_local[index] + preference_local[index])
    var neutral_goal = List[Float32](capacity=n_states)
    for _ in range(n_states):
        neutral_goal.append(1.0 / Float32(n_states))
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
        combined,
        List[Float32](),
        neutral_goal,
        action_prior,
        horizon,
        n_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        0,
        0,
        List[Float32](),
        False,
        False,
    )


def active_inference_planning_dense(
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
    debug_assert(
        len(transition_tensor) == n_states * n_states * n_static * n_actions,
        "transition shape mismatch",
    )
    var log_prior = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_base = _compute_dense_log_base(
        log_transition,
        log_prior,
        n_states,
        n_actions,
        n_static,
    )
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
        List[Float32](),
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
        log_transition,
        True,
        False,
    )


def active_inference_planning_dense_theta_goal(
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
    debug_assert(
        len(transition_tensor) == n_states * n_states * n_static * n_actions,
        "transition shape mismatch",
    )
    var log_prior = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_base = _compute_dense_log_base(
        log_transition,
        log_prior,
        n_states,
        n_actions,
        n_static,
    )
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
        List[Float32](),
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
        log_transition,
        True,
        True,
    )


def active_inference_planning_dense_until_converged(
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
    """Dense AIF-MP with residual early stopping and optional adaptation.

    Returns the normal action/dynamics/observation-channel payload followed by
    `[converged, iterations, final_residual, final_damping, residuals...]`.
    """
    debug_assert(
        len(transition_tensor) == n_states * n_states * n_static * n_actions,
        "transition shape mismatch",
    )
    var log_prior = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_base = _compute_dense_log_base(
        log_transition,
        log_prior,
        n_states,
        n_actions,
        n_static,
    )
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
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
        log_transition,
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


struct PreparedDenseAIFMP[horizon: Int, iterations: Int]:
    """Compile-time-specialized dense AIF-MP with static logs cached."""

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
        var log_prior = _safe_log_values(q_static_state)
        var log_base = _compute_dense_log_base(
            self.log_transition,
            log_prior,
            self.n_states,
            self.n_actions,
            self.n_static,
        )
        return _active_inference_planning(
            q_current_state,
            q_static_state,
            log_base,
            List[Float32](),
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
            self.log_transition,
            True,
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


def active_inference_planning_dense_specialized[
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
    """Specialize dense AIF-MP on horizon and iteration parameters."""
    var log_prior = _safe_log_values(q_static_state)
    var log_transition = _safe_log_values(transition_tensor)
    var log_base = _compute_dense_log_base(
        log_transition, log_prior, n_states, n_actions, n_static
    )
    return _active_inference_planning(
        q_current_state,
        q_static_state,
        log_base,
        List[Float32](),
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
        log_transition,
        True,
        theta_goal,
    )
