from std.collections import List

from aif_mojo.dyn_channel_loopy_bp import (
    _add_messages,
    _compute_pref_to_x,
    _damp_dyn_channels,
    _dense_dyn_channels_and_pair,
    _dense_dyn_region_beliefs,
    _initial_dense_dyn_channels,
    _dense_reduced_dyn_channel,
    _safe_log_values,
    _zeros,
)
from aif_mojo.numerics import safe_log
from aif_mojo.precise_info_seeking import (
    _backward_pass_precise,
    _compute_precise_obs_channels,
    _compute_precise_obs_regions,
    _compute_precise_obs_to_x,
    _damp_obs_channels,
    _forward_pass_precise,
    _initial_precise_obs_channels,
    compute_precise_obs_kernels,
)


def _region_extended_loopy_bp_planning(
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
    """Dense region-extended planner with dynamic and observation channels.

    The result contains the first-step action distribution, then log
    r(new_state | old_state, action) in (time, old, new, action) order,
    then log r(observation | state, theta) in
    (time, fov, observation, state, theta) order.
    """
    debug_assert(
        len(q_current_state) == n_states, "current state shape mismatch"
    )
    debug_assert(len(q_static_state) == n_static, "static state shape mismatch")
    debug_assert(
        len(transition_tensor) == n_states * n_states * n_static * n_actions,
        "transition tensor shape mismatch",
    )
    debug_assert(
        len(observation_tensor) == n_fov * n_obs_types * n_states * n_static,
        "observation tensor shape mismatch",
    )
    debug_assert(len(action_prior) == n_actions, "action prior shape mismatch")
    if theta_goal:
        debug_assert(
            len(goal) == n_states * n_static, "theta goal shape mismatch"
        )
    else:
        debug_assert(len(goal) == n_states, "goal shape mismatch")

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

    var log_dyn_channels = List[Float32]()
    var initial_dyn_channel = -safe_log(Float32(n_states))
    for _ in range(horizon * n_states * n_states * n_actions):
        log_dyn_channels.append(initial_dyn_channel)

    var log_obs_channels = List[Float32]()
    var initial_obs_channel = -safe_log(Float32(n_obs_types))
    for _ in range((horizon + 1) * n_fov * n_obs_types * n_states * n_static):
        log_obs_channels.append(initial_obs_channel)
    var log_marginal_obs_channels = List[Float32]()
    for _ in range((horizon + 1) * n_fov * n_obs_types * n_states):
        log_marginal_obs_channels.append(initial_obs_channel)
    if convergence_mode:
        log_dyn_channels = _initial_dense_dyn_channels(
            log_transition,
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

    # Region-extended dynamics divide by r(new | old, action), but do not
    # multiply by a learned action channel.
    var unit_action_channels = _zeros(horizon * n_states * n_actions)
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
            log_dyn_channels,
            unit_action_channels,
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

        var dyn_result = _dense_dyn_channels_and_pair(
            log_transition,
            log_fwd,
            log_bwd,
            log_local_to_x,
            log_prior_dyn,
            log_action_prior,
            log_dyn_channels,
            unit_action_channels,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var channel_size = horizon * n_states * n_states * n_actions
        var raw_dyn_channels = List[Float32]()
        for index in range(channel_size):
            raw_dyn_channels.append(dyn_result[index])

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
            var log_dyn_regions = _dense_dyn_region_beliefs(
                log_transition,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_prior_dyn,
                log_action_prior,
                log_dyn_channels,
                unit_action_channels,
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
            for value in log_dyn_regions:
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

        log_dyn_channels = _damp_dyn_channels(
            log_dyn_channels,
            raw_dyn_channels,
            damping,
            horizon,
            n_states,
            n_actions,
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
    for value in log_dyn_channels:
        result.append(value)
    for value in log_obs_channels:
        result.append(value)
    if record_history:
        for value in diagnostic_history:
            result.append(value)
    return result^


def region_extended_loopy_bp_planning_dense(
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
    """Plan with a terminal state goal."""
    return _region_extended_loopy_bp_planning(
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


def region_extended_loopy_bp_planning_dense_theta_goal(
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
    """Plan with per-step preferences in (state, theta) order."""
    return _region_extended_loopy_bp_planning(
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
