from std.collections import List

from aif_mojo.dyn_channel_loopy_bp import (
    _add_messages,
    _backward_pass,
    _compute_obs_to_theta,
    _compute_obs_to_x,
    _compute_pref_to_theta,
    _compute_pref_to_x,
    _compute_theta_cavities_extended,
    _damp_action_channels,
    _dense_dyn_channels_and_pair,
    _dense_dyn_region_beliefs,
    _dense_dyn_to_theta_dyn_channel,
    _dense_reduced_dyn_channel,
    _forward_pass,
    _normalize_action_channels,
    _safe_log_values,
    _zeros,
)
from aif_mojo.numerics import safe_log
from aif_mojo.convergence_control import (
    append_convergence_metadata,
    max_channel_residual,
    next_adaptive_damping,
)
from aif_mojo.sparse_messages import (
    sparse_dyn_to_theta_weighted,
    sparse_pair_marginal_weighted,
    sparse_reduced_weighted,
)


def _log_static_input(
    values: List[Float32], already_log: Bool
) -> List[Float32]:
    if already_log:
        return values.copy()
    return _safe_log_values(values)


def _reduce_vbp_transition(
    use_sparse: Bool,
    transition_indices: List[Int],
    log_transition: List[Float32],
    log_cavity: List[Float32],
    unit_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    if use_sparse:
        return sparse_reduced_weighted(
            transition_indices,
            log_cavity,
            log_action_channel,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
    return _dense_reduced_dyn_channel(
        log_transition,
        log_cavity,
        unit_dyn_channels,
        log_action_channel,
        horizon,
        n_states,
        n_actions,
        n_static,
    )


def _vbp_channel_planning(
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
    convergence_tolerance: Float32 = -1.0,
    minimum_iterations: Int = 1,
    record_residuals: Bool = False,
    adaptive_damping: Bool = False,
    minimum_damping: Float32 = 0.05,
    maximum_damping: Float32 = 1.0,
    static_inputs_are_log: Bool = False,
) -> List[Float32]:
    """Shared dense/sparse VBP action-channel planner."""
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

    var log_q0 = _safe_log_values(q_current_state)
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_transition = _log_static_input(
        transition_tensor, static_inputs_are_log
    )
    var log_observation = _log_static_input(
        observation_tensor, static_inputs_are_log
    )
    var log_preference = _log_static_input(goal, static_inputs_are_log)
    var log_goal = _zeros(n_states)
    if not theta_goal:
        log_goal = _log_static_input(goal, static_inputs_are_log)
    var log_action_prior = _log_static_input(
        action_prior, static_inputs_are_log
    )
    var log_action_per_t = List[Float32]()
    for _ in range(horizon):
        for action_idx in range(n_actions):
            log_action_per_t.append(log_action_prior[action_idx])

    var log_dyn_to_theta = _zeros(horizon * n_static)
    var log_obs_to_theta = _zeros((horizon + 1) * n_static)
    var log_pref_to_theta = _zeros((horizon + 1) * n_static)
    var q_u = _zeros(horizon * n_actions)
    var log_action_channel = List[Float32]()
    var initial_action_channel = -safe_log(Float32(n_actions))
    for _ in range(horizon * n_states * n_actions):
        log_action_channel.append(initial_action_channel)
    var unit_dyn_channels = _zeros(horizon * n_states * n_states * n_actions)
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

        var reduced = _reduce_vbp_transition(
            use_sparse,
            transition_indices,
            log_transition,
            log_cavity_dyn,
            unit_dyn_channels,
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
            log_dyn_to_theta = sparse_dyn_to_theta_weighted(
                transition_indices,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_action_per_t,
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
                unit_dyn_channels,
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

        var log_pair = List[Float32]()
        if use_sparse:
            log_pair = sparse_pair_marginal_weighted(
                transition_indices,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_cavity_dyn,
                log_action_prior,
                log_action_channel,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        else:
            var channel_and_pair = _dense_dyn_channels_and_pair(
                log_transition,
                log_fwd,
                log_bwd,
                log_local_to_x,
                log_cavity_dyn,
                log_action_prior,
                unit_dyn_channels,
                log_action_channel,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
            var channel_size = horizon * n_states * n_states * n_actions
            for index in range(horizon * n_states * n_actions):
                log_pair.append(channel_and_pair[channel_size + index])
        var raw_action_channel = _normalize_action_channels(
            log_pair, horizon, n_states, n_actions
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
                unit_dyn_channels,
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
        var damped_action_channel = _damp_action_channels(
            log_action_channel,
            raw_action_channel,
            effective_damping,
            horizon,
            n_states,
            n_actions,
        )
        final_residual = max_channel_residual(
            log_action_channel, damped_action_channel
        )
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
    for value in log_action_channel:
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


def vbp_channel_planning_sparse(
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
    var no_transition_tensor = List[Float32]()
    return _vbp_channel_planning(
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


def vbp_channel_planning_sparse_theta_goal(
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
    var no_transition_tensor = List[Float32]()
    return _vbp_channel_planning(
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


def vbp_channel_planning_dense(
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
    var no_transition_indices = List[Int]()
    return _vbp_channel_planning(
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


def vbp_channel_planning_dense_theta_goal(
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
    var no_transition_indices = List[Int]()
    return _vbp_channel_planning(
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


def vbp_channel_planning_dense_until_converged(
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
    """Dense VBP with channel-residual early stopping.

    Returns the normal action/channel payload followed by
    `[converged, iterations, final_residual, final_damping, residuals...]`.
    """
    var no_transition_indices = List[Int]()
    return _vbp_channel_planning(
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
        tolerance,
        minimum_iterations,
        True,
        adaptive_damping,
        minimum_damping,
        maximum_damping,
    )


def vbp_channel_planning_sparse_until_converged(
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
    """Sparse VBP with the same residual contract as the dense API."""
    return _vbp_channel_planning(
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
        tolerance,
        minimum_iterations,
        True,
        adaptive_damping,
        minimum_damping,
        maximum_damping,
    )


struct PreparedDenseVBP[horizon: Int, iterations: Int]:
    """Compile-time-specialized VBP with static log inputs cached."""

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
        return _vbp_channel_planning(
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
            -1.0,
            1,
            False,
            False,
            0.05,
            1.0,
            True,
        )


def vbp_channel_planning_dense_specialized[
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
    """Specialize VBP on a compile-time horizon and iteration budget."""
    return _vbp_channel_planning(
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
