from std.collections import List
from std.math import exp

from aif_mojo.active_inference import (
    _active_inference_planning,
    _compute_dense_log_base,
)
from aif_mojo.loopy_bp import (
    _add_messages as _loopy_add_messages,
    _backward_messages_with_local,
    _forward_pass_with_local,
    backward_messages,
    compute_action_marginals,
    compute_dyn_to_theta_dense,
    compute_reduced_per_t_dense,
    compute_theta_cavities,
    forward_pass,
)
from aif_mojo.dyn_channel_loopy_bp import _dyn_channel_loopy_bp_planning
from aif_mojo.nuijten_mp import _nuijten_mp_planning
from aif_mojo.numerics import LOG_ZERO_THRESHOLD, logsumexp, safe_log
from aif_mojo.precise_info_seeking import _precise_info_seeking_planning
from aif_mojo.region_extended_loopy_bp import (
    _region_extended_loopy_bp_planning,
)
from aif_mojo.vbp_channel import _vbp_channel_planning


def entropy(log_q: List[Float32]) -> Float32:
    """Shannon entropy of normalized log probabilities."""
    var result = Float32(0.0)
    for value in log_q:
        var probability = exp(value)
        if probability > 1.0e-30:
            result -= probability * value
    return result


def entropy_unnormalized(log_belief: List[Float32]) -> Float32:
    """Shannon entropy after normalizing a flat log belief."""
    var normalizer = logsumexp(log_belief)
    var result = Float32(0.0)
    for value in log_belief:
        var normalized = value - normalizer
        var probability = exp(normalized)
        if probability > 1.0e-30:
            result -= probability * normalized
    return result


def energy(log_region: List[Float32], log_factor: List[Float32]) -> Float32:
    """Average factor energy under a normalized flat region belief."""
    debug_assert(len(log_region) == len(log_factor), "factor shape mismatch")
    var normalizer = logsumexp(log_region)
    var result = Float32(0.0)
    for i in range(len(log_region)):
        if log_factor[i] > LOG_ZERO_THRESHOLD:
            result -= exp(log_region[i] - normalizer) * log_factor[i]
    return result


def _dyn_offset(
    time_idx: Int,
    old_idx: Int,
    new_idx: Int,
    static_idx: Int,
    action_idx: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> Int:
    return (
        ((time_idx * n_states + old_idx) * n_states + new_idx) * n_static
        + static_idx
    ) * n_actions + action_idx


def _obs_offset(
    time_idx: Int,
    fov_idx: Int,
    observation_idx: Int,
    state_idx: Int,
    static_idx: Int,
    n_fov: Int,
    n_observations: Int,
    n_states: Int,
    n_static: Int,
) -> Int:
    return (
        ((time_idx * n_fov + fov_idx) * n_observations + observation_idx)
        * n_states
        + state_idx
    ) * n_static + static_idx


def action_cond_entropy(
    log_dyn_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> Float32:
    """Map JAX `_action_cond_entropy`: sum_t H[q(u|x)]."""
    debug_assert(
        len(log_dyn_regions)
        == horizon * n_states * n_states * n_static * n_actions,
        "dynamics region shape mismatch",
    )
    var result = Float32(0.0)
    for time_idx in range(horizon):
        var log_pair = List[Float32]()
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var terms = List[Float32]()
                for new_idx in range(n_states):
                    for static_idx in range(n_static):
                        terms.append(
                            log_dyn_regions[
                                _dyn_offset(
                                    time_idx,
                                    old_idx,
                                    new_idx,
                                    static_idx,
                                    action_idx,
                                    n_states,
                                    n_static,
                                    n_actions,
                                )
                            ]
                        )
                log_pair.append(logsumexp(terms))
        var log_x = List[Float32]()
        for old_idx in range(n_states):
            var terms = List[Float32]()
            for action_idx in range(n_actions):
                terms.append(log_pair[old_idx * n_actions + action_idx])
            log_x.append(logsumexp(terms))
        result += entropy_unnormalized(log_pair) - entropy_unnormalized(log_x)
    return result


def dyn_cond_entropy(
    log_dyn_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> Float32:
    """Map JAX `_dyn_cond_entropy`: sum_t H[q(x'|x,u)]."""
    debug_assert(
        len(log_dyn_regions)
        == horizon * n_states * n_states * n_static * n_actions,
        "dynamics region shape mismatch",
    )
    var result = Float32(0.0)
    for time_idx in range(horizon):
        var log_xxu = List[Float32]()
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var terms = List[Float32]()
                    for static_idx in range(n_static):
                        terms.append(
                            log_dyn_regions[
                                _dyn_offset(
                                    time_idx,
                                    old_idx,
                                    new_idx,
                                    static_idx,
                                    action_idx,
                                    n_states,
                                    n_static,
                                    n_actions,
                                )
                            ]
                        )
                    log_xxu.append(logsumexp(terms))
        var log_xu = List[Float32]()
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var terms = List[Float32]()
                for new_idx in range(n_states):
                    var offset = (
                        old_idx * n_states + new_idx
                    ) * n_actions + action_idx
                    terms.append(log_xxu[offset])
                log_xu.append(logsumexp(terms))
        result += entropy_unnormalized(log_xxu) - entropy_unnormalized(log_xu)
    return result


def obs_cond_entropy(
    log_obs_regions: List[Float32],
    horizon: Int,
    n_fov: Int,
    n_observations: Int,
    n_states: Int,
    n_static: Int,
) -> Float32:
    """Map JAX `_obs_cond_entropy`, including its factor-of-two correction."""
    debug_assert(
        len(log_obs_regions)
        == (horizon + 1) * n_fov * n_observations * n_states * n_static,
        "observation region shape mismatch",
    )
    var result = Float32(0.0)
    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            var region = List[Float32]()
            for observation_idx in range(n_observations):
                for state_idx in range(n_states):
                    for static_idx in range(n_static):
                        region.append(
                            log_obs_regions[
                                _obs_offset(
                                    time_idx,
                                    fov_idx,
                                    observation_idx,
                                    state_idx,
                                    static_idx,
                                    n_fov,
                                    n_observations,
                                    n_states,
                                    n_static,
                                )
                            ]
                        )
            var log_xtheta = List[Float32]()
            for state_idx in range(n_states):
                for static_idx in range(n_static):
                    var terms = List[Float32]()
                    for observation_idx in range(n_observations):
                        terms.append(
                            region[
                                (observation_idx * n_states + state_idx)
                                * n_static
                                + static_idx
                            ]
                        )
                    log_xtheta.append(logsumexp(terms))
            var log_yx = List[Float32]()
            for observation_idx in range(n_observations):
                for state_idx in range(n_states):
                    var terms = List[Float32]()
                    for static_idx in range(n_static):
                        terms.append(
                            region[
                                (observation_idx * n_states + state_idx)
                                * n_static
                                + static_idx
                            ]
                        )
                    log_yx.append(logsumexp(terms))
            var log_x = List[Float32]()
            for state_idx in range(n_states):
                var terms = List[Float32]()
                for observation_idx in range(n_observations):
                    terms.append(log_yx[observation_idx * n_states + state_idx])
                log_x.append(logsumexp(terms))
            var h_y_given_xtheta = entropy_unnormalized(
                region
            ) - entropy_unnormalized(log_xtheta)
            var h_y_given_x = entropy_unnormalized(
                log_yx
            ) - entropy_unnormalized(log_x)
            result += 2.0 * h_y_given_xtheta - h_y_given_x
    return result


def compute_bethe_vfe_loopy(
    log_transition: List[Float32],
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    q_u: List[Float32],
    log_cavity_theta: List[Float32],
    log_prior_theta: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> Float32:
    """Map JAX `compute_bethe_vfe_loopy` for flat dense tensors."""
    debug_assert(horizon > 0, "horizon must be positive")
    debug_assert(
        len(log_transition) == n_states * n_states * n_static * n_actions,
        "transition shape mismatch",
    )
    debug_assert(
        len(log_reduced_per_t) == horizon * n_states * n_states * n_actions,
        "reduced transition shape mismatch",
    )
    debug_assert(
        len(log_fwd_messages) == (horizon + 1) * n_states,
        "forward message shape mismatch",
    )
    debug_assert(
        len(log_bwd_messages) == (horizon + 1) * n_states,
        "backward message shape mismatch",
    )
    debug_assert(len(q_u) == horizon * n_actions, "action shape mismatch")
    debug_assert(
        len(log_cavity_theta) == horizon * n_static,
        "theta cavity shape mismatch",
    )
    debug_assert(len(log_prior_theta) == n_static, "theta prior shape mismatch")
    debug_assert(
        len(log_action_prior) == n_actions,
        "action prior shape mismatch",
    )

    var factor_vfe = Float32(0.0)
    for time_idx in range(horizon):
        var log_region = List[Float32]()
        var log_factor = List[Float32]()
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for static_idx in range(n_static):
                    for action_idx in range(n_actions):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var factor = log_transition[transition_offset]
                        log_factor.append(factor)
                        log_region.append(
                            factor
                            + log_fwd_messages[time_idx * n_states + old_idx]
                            + log_bwd_messages[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_cavity_theta[time_idx * n_static + static_idx]
                            + safe_log(q_u[time_idx * n_actions + action_idx])
                        )
        factor_vfe += energy(log_region, log_factor)
        factor_vfe -= entropy_unnormalized(log_region)

    var singleton_entropy = Float32(0.0)
    for time_idx in range(1, horizon):
        var log_q_x = List[Float32]()
        for state_idx in range(n_states):
            log_q_x.append(
                log_fwd_messages[time_idx * n_states + state_idx]
                + log_bwd_messages[time_idx * n_states + state_idx]
            )
        var normalizer = logsumexp(log_q_x)
        for state_idx in range(n_states):
            log_q_x[state_idx] -= normalizer
        singleton_entropy += entropy(log_q_x)

    var log_q_theta = List[Float32]()
    for static_idx in range(n_static):
        var value = Float32(0.0)
        for time_idx in range(horizon):
            value += log_cavity_theta[time_idx * n_static + static_idx]
        log_q_theta.append(value / Float32(horizon))
    var theta_normalizer = logsumexp(log_q_theta)
    for static_idx in range(n_static):
        log_q_theta[static_idx] -= theta_normalizer
    var theta_term = Float32(1 - horizon) * entropy(log_q_theta)
    return factor_vfe - singleton_entropy - theta_term


def compute_region_extended_vfe(
    log_transition: List[Float32],
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    q_u: List[Float32],
    log_cavity_dyn: List[Float32],
    log_prior_theta: List[Float32],
    log_action_prior: List[Float32],
    log_dyn_regions: List[Float32],
    log_obs_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_fov: Int,
    n_observations: Int,
) -> Float32:
    var base = compute_bethe_vfe_loopy(
        log_transition,
        log_reduced_per_t,
        log_fwd_messages,
        log_bwd_messages,
        q_u,
        log_cavity_dyn,
        log_prior_theta,
        log_action_prior,
        horizon,
        n_states,
        n_static,
        n_actions,
    )
    return (
        base
        + obs_cond_entropy(
            log_obs_regions,
            horizon,
            n_fov,
            n_observations,
            n_states,
            n_static,
        )
        - dyn_cond_entropy(
            log_dyn_regions, horizon, n_states, n_static, n_actions
        )
    )


def compute_dyn_channel_vfe(
    log_transition: List[Float32],
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    q_u: List[Float32],
    log_cavity_dyn: List[Float32],
    log_prior_theta: List[Float32],
    log_action_prior: List[Float32],
    log_dyn_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> Float32:
    return compute_bethe_vfe_loopy(
        log_transition,
        log_reduced_per_t,
        log_fwd_messages,
        log_bwd_messages,
        q_u,
        log_cavity_dyn,
        log_prior_theta,
        log_action_prior,
        horizon,
        n_states,
        n_static,
        n_actions,
    ) - dyn_cond_entropy(
        log_dyn_regions, horizon, n_states, n_static, n_actions
    )


def compute_vbp_channel_vfe(
    log_transition: List[Float32],
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    q_u: List[Float32],
    log_cavity_dyn: List[Float32],
    log_prior_theta: List[Float32],
    log_action_prior: List[Float32],
    log_dyn_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> Float32:
    return compute_bethe_vfe_loopy(
        log_transition,
        log_reduced_per_t,
        log_fwd_messages,
        log_bwd_messages,
        q_u,
        log_cavity_dyn,
        log_prior_theta,
        log_action_prior,
        horizon,
        n_states,
        n_static,
        n_actions,
    ) + action_cond_entropy(
        log_dyn_regions, horizon, n_states, n_static, n_actions
    )


def compute_precise_info_seeking_vfe(
    log_transition: List[Float32],
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    q_u: List[Float32],
    log_cavity_dyn: List[Float32],
    log_prior_theta: List[Float32],
    log_action_prior: List[Float32],
    log_dyn_regions: List[Float32],
    log_obs_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_fov: Int,
    n_observations: Int,
) -> Float32:
    var base = compute_bethe_vfe_loopy(
        log_transition,
        log_reduced_per_t,
        log_fwd_messages,
        log_bwd_messages,
        q_u,
        log_cavity_dyn,
        log_prior_theta,
        log_action_prior,
        horizon,
        n_states,
        n_static,
        n_actions,
    )
    return (
        base
        + action_cond_entropy(
            log_dyn_regions, horizon, n_states, n_static, n_actions
        )
        + obs_cond_entropy(
            log_obs_regions,
            horizon,
            n_fov,
            n_observations,
            n_states,
            n_static,
        )
    )


def compute_active_inference_vfe(
    log_transition: List[Float32],
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    q_u: List[Float32],
    log_cavity_dyn: List[Float32],
    log_prior_theta: List[Float32],
    log_action_prior: List[Float32],
    log_dyn_regions: List[Float32],
    log_obs_regions: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_fov: Int,
    n_observations: Int,
) -> Float32:
    return compute_precise_info_seeking_vfe(
        log_transition,
        log_reduced_per_t,
        log_fwd_messages,
        log_bwd_messages,
        q_u,
        log_cavity_dyn,
        log_prior_theta,
        log_action_prior,
        log_dyn_regions,
        log_obs_regions,
        horizon,
        n_states,
        n_static,
        n_actions,
        n_fov,
        n_observations,
    ) - dyn_cond_entropy(
        log_dyn_regions, horizon, n_states, n_static, n_actions
    )


def compute_nuijten_vfe(
    log_dyn_regions: List[Float32],
    obs_regions: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    q_u: List[Float32],
    log_transition_kernel_tiled: List[Float32],
    log_observation_factor: List[Float32],
    log_cavity_dyn: List[Float32],
    log_cavity_obs: List[Float32],
    log_prior_theta: List[Float32],
    action_prior_per_t: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_fov: Int,
    n_observations: Int,
) -> Float32:
    """Map JAX `compute_nuijten_vfe`; obs regions remain probability-space."""
    debug_assert(
        len(log_transition_kernel_tiled) == len(log_dyn_regions),
        "transition kernel shape mismatch",
    )
    debug_assert(
        len(obs_regions)
        == (horizon + 1) * n_fov * n_observations * n_states * n_static,
        "observation region shape mismatch",
    )
    debug_assert(
        len(log_observation_factor)
        == n_fov * n_observations * n_states * n_static,
        "observation factor shape mismatch",
    )
    debug_assert(len(log_prior_theta) == n_static, "theta prior shape mismatch")

    var dynamic_energy = Float32(0.0)
    var dyn_region_size = n_states * n_states * n_static * n_actions
    for time_idx in range(horizon):
        var region = List[Float32]()
        var factor = List[Float32]()
        for offset in range(dyn_region_size):
            region.append(log_dyn_regions[time_idx * dyn_region_size + offset])
            factor.append(
                log_transition_kernel_tiled[time_idx * dyn_region_size + offset]
            )
        dynamic_energy += energy(region, factor)

    var log_obs_regions = List[Float32]()
    for value in obs_regions:
        log_obs_regions.append(safe_log(value))
    var observation_energy = Float32(0.0)
    var obs_region_size = n_observations * n_states * n_static
    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            var region = List[Float32]()
            var factor = List[Float32]()
            var region_base = (time_idx * n_fov + fov_idx) * obs_region_size
            var factor_base = fov_idx * obs_region_size
            for offset in range(obs_region_size):
                region.append(log_obs_regions[region_base + offset])
                factor.append(log_observation_factor[factor_base + offset])
            observation_energy += energy(region, factor)

    var prior_energy = Float32(0.0)
    for time_idx in range(horizon):
        for action_idx in range(n_actions):
            prior_energy -= q_u[time_idx * n_actions + action_idx] * safe_log(
                action_prior_per_t[time_idx * n_actions + action_idx]
            )

    var state_entropy = Float32(0.0)
    for time_idx in range(horizon + 1):
        var log_q_x = List[Float32]()
        for state_idx in range(n_states):
            log_q_x.append(
                log_fwd_messages[time_idx * n_states + state_idx]
                + log_bwd_messages[time_idx * n_states + state_idx]
            )
        var normalizer = logsumexp(log_q_x)
        for state_idx in range(n_states):
            log_q_x[state_idx] -= normalizer
        state_entropy += entropy(log_q_x)

    var action_entropy = Float32(0.0)
    for time_idx in range(horizon):
        var log_q_u = List[Float32]()
        for action_idx in range(n_actions):
            log_q_u.append(safe_log(q_u[time_idx * n_actions + action_idx]))
        action_entropy += entropy(log_q_u)

    var log_q_theta = List[Float32]()
    var cavity_count = 2 * horizon + 1
    for static_idx in range(n_static):
        var value = Float32(0.0)
        for time_idx in range(horizon):
            value += log_cavity_dyn[time_idx * n_static + static_idx]
        for time_idx in range(horizon + 1):
            value += log_cavity_obs[time_idx * n_static + static_idx]
        log_q_theta.append(value / Float32(cavity_count))
    var theta_normalizer = logsumexp(log_q_theta)
    for static_idx in range(n_static):
        log_q_theta[static_idx] -= theta_normalizer
    var theta_entropy = entropy(log_q_theta)

    var dynamic_intersection = Float32(0.0)
    for time_idx in range(horizon):
        var log_xu = List[Float32]()
        var log_xxu = List[Float32]()
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var xu_terms = List[Float32]()
                for new_idx in range(n_states):
                    for static_idx in range(n_static):
                        xu_terms.append(
                            log_dyn_regions[
                                _dyn_offset(
                                    time_idx,
                                    old_idx,
                                    new_idx,
                                    static_idx,
                                    action_idx,
                                    n_states,
                                    n_static,
                                    n_actions,
                                )
                            ]
                        )
                log_xu.append(logsumexp(xu_terms))
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var xxu_terms = List[Float32]()
                    for static_idx in range(n_static):
                        xxu_terms.append(
                            log_dyn_regions[
                                _dyn_offset(
                                    time_idx,
                                    old_idx,
                                    new_idx,
                                    static_idx,
                                    action_idx,
                                    n_states,
                                    n_static,
                                    n_actions,
                                )
                            ]
                        )
                    log_xxu.append(logsumexp(xxu_terms))
        dynamic_intersection += entropy_unnormalized(
            log_xu
        ) - entropy_unnormalized(log_xxu)

    var observation_intersection = Float32(0.0)
    for time_idx in range(horizon + 1):
        for fov_idx in range(n_fov):
            var region = List[Float32]()
            var log_xtheta = List[Float32]()
            for observation_idx in range(n_observations):
                for state_idx in range(n_states):
                    for static_idx in range(n_static):
                        region.append(
                            log_obs_regions[
                                _obs_offset(
                                    time_idx,
                                    fov_idx,
                                    observation_idx,
                                    state_idx,
                                    static_idx,
                                    n_fov,
                                    n_observations,
                                    n_states,
                                    n_static,
                                )
                            ]
                        )
            for state_idx in range(n_states):
                for static_idx in range(n_static):
                    var terms = List[Float32]()
                    for observation_idx in range(n_observations):
                        terms.append(
                            region[
                                (observation_idx * n_states + state_idx)
                                * n_static
                                + static_idx
                            ]
                        )
                    log_xtheta.append(logsumexp(terms))
            observation_intersection += entropy_unnormalized(
                region
            ) - entropy_unnormalized(log_xtheta)

    return (
        dynamic_energy
        + observation_energy
        + prior_energy
        - (state_entropy + action_entropy + theta_entropy)
        + dynamic_intersection
        + observation_intersection
    )


def _trace_zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def _trace_safe_log(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
    return result^


def _validate_trace_goal_shape(
    goal: List[Float32], theta_goal: Bool, n_states: Int, n_static: Int
):
    if theta_goal:
        debug_assert(
            len(goal) == n_states * n_static, "theta goal shape mismatch"
        )
    else:
        debug_assert(len(goal) == n_states, "terminal goal shape mismatch")


def loopy_bp_planning_with_vfe(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    theta_goal: Bool,
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense JAX `loopy_bp_convergence`; returns action then VFE trace."""
    _validate_trace_goal_shape(goal, theta_goal, n_states, n_static)
    var log_transition = _trace_safe_log(transition_tensor)
    var log_prior_theta = _trace_safe_log(q_static_state)
    var log_q0 = _trace_safe_log(q_current_state)
    var log_action_prior = _trace_safe_log(action_prior)
    var log_goal = _trace_safe_log(goal)
    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for value in log_prior_theta:
            log_cavity_theta.append(value)
    var previous_dyn_to_theta = _trace_zeros(horizon * n_static)
    var q_u = _trace_zeros(horizon * n_actions)
    var trace = List[Float32]()

    for _ in range(n_iterations):
        var reduced = compute_reduced_per_t_dense(
            log_transition,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var log_fwd: List[Float32]
        var log_bwd: List[Float32]
        if theta_goal:
            var theta_logits = List[Float32]()
            for static_idx in range(n_static):
                var value = log_prior_theta[static_idx]
                for time_idx in range(horizon):
                    value += previous_dyn_to_theta[
                        time_idx * n_static + static_idx
                    ]
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
                for value in log_preference:
                    local_messages.append(value)

            log_fwd = _forward_pass_with_local(
                reduced,
                log_q0,
                log_action_prior,
                local_messages,
                horizon,
                n_states,
                n_actions,
            )
            log_bwd = _backward_messages_with_local(
                reduced,
                log_action_prior,
                local_messages,
                horizon,
                n_states,
                n_actions,
            )
            var fwd_with_local = _loopy_add_messages(log_fwd, local_messages)
            var bwd_with_local = _loopy_add_messages(log_bwd, local_messages)
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
        else:
            log_fwd = forward_pass(
                reduced,
                log_q0,
                log_action_prior,
                horizon,
                n_states,
                n_actions,
            )
            log_bwd = backward_messages(
                reduced,
                log_goal,
                log_action_prior,
                horizon,
                n_states,
                n_actions,
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
            previous_dyn_to_theta = compute_dyn_to_theta_dense(
                log_transition,
                log_fwd,
                log_bwd,
                log_action_prior,
                horizon,
                n_states,
                n_actions,
                n_static,
            )
        trace.append(
            compute_bethe_vfe_loopy(
                log_transition,
                reduced,
                log_fwd,
                log_bwd,
                q_u,
                log_cavity_theta,
                log_prior_theta,
                log_action_prior,
                horizon,
                n_states,
                n_static,
                n_actions,
            )
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta,
            previous_dyn_to_theta,
            horizon,
            n_static,
        )

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    for value in trace:
        result.append(value)
    return result^


def _trace_slice(
    values: List[Float32], start: Int, length: Int
) -> List[Float32]:
    var result = List[Float32]()
    for index in range(length):
        result.append(values[start + index])
    return result^


def _trace_tiled_prior(log_prior: List[Float32], count: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(count):
        for value in log_prior:
            result.append(value)
    return result^


def _trace_transition_kernel_tiled(
    log_transition: List[Float32],
    horizon: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for static_idx in range(n_static):
                    for action_idx in range(n_actions):
                        var offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        result.append(log_transition[offset])
    return result^


def region_extended_loopy_bp_planning_with_vfe(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    theta_goal: Bool,
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense JAX `region_extended_convergence`, flattened with trace last."""
    _validate_trace_goal_shape(goal, theta_goal, n_states, n_static)
    var diagnostic = _region_extended_loopy_bp_planning(
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
        theta_goal,
        True,
        True,
    )
    var dyn_channel_size = horizon * n_states * n_states * n_actions
    var obs_channel_size = (
        (horizon + 1) * n_fov * n_obs_types * n_states * n_static
    )
    var prefix_size = n_actions + dyn_channel_size + obs_channel_size
    var reduced_size = horizon * n_states * n_states * n_actions
    var message_size = (horizon + 1) * n_states
    var action_size = horizon * n_actions
    var dyn_region_size = horizon * n_states * n_states * n_static * n_actions
    var history_size = (
        reduced_size
        + 2 * message_size
        + action_size
        + dyn_region_size
        + obs_channel_size
    )
    var log_transition = _trace_safe_log(transition_tensor)
    var log_prior_theta = _trace_safe_log(q_static_state)
    var log_action_prior = _trace_safe_log(action_prior)
    var log_prior_dyn = _trace_tiled_prior(log_prior_theta, horizon)
    var trace = List[Float32]()
    for iteration in range(n_iterations):
        var cursor = prefix_size + iteration * history_size
        var reduced = _trace_slice(diagnostic, cursor, reduced_size)
        cursor += reduced_size
        var log_fwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var log_bwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var q_u = _trace_slice(diagnostic, cursor, action_size)
        cursor += action_size
        var dyn_regions = _trace_slice(diagnostic, cursor, dyn_region_size)
        cursor += dyn_region_size
        var obs_regions = _trace_slice(diagnostic, cursor, obs_channel_size)
        trace.append(
            compute_region_extended_vfe(
                log_transition,
                reduced,
                log_fwd,
                log_bwd,
                q_u,
                log_prior_dyn,
                log_prior_theta,
                log_action_prior,
                dyn_regions,
                obs_regions,
                horizon,
                n_states,
                n_static,
                n_actions,
                n_fov,
                n_obs_types,
            )
        )
    var result = _trace_slice(diagnostic, 0, prefix_size)
    for value in trace:
        result.append(value)
    return result^


def dyn_channel_loopy_bp_planning_with_vfe(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    theta_goal: Bool,
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense JAX `dyn_channel_convergence`, flattened with trace last."""
    var no_indices = List[Int]()
    _validate_trace_goal_shape(goal, theta_goal, n_states, n_static)
    var diagnostic = _dyn_channel_loopy_bp_planning(
        q_current_state,
        q_static_state,
        no_indices,
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
        theta_goal,
        True,
        True,
    )
    var channel_size = horizon * n_states * n_states * n_actions
    var prefix_size = n_actions + channel_size
    var reduced_size = channel_size
    var message_size = (horizon + 1) * n_states
    var action_size = horizon * n_actions
    var cavity_size = horizon * n_static
    var dyn_region_size = channel_size * n_static
    var history_size = (
        reduced_size
        + 2 * message_size
        + action_size
        + cavity_size
        + dyn_region_size
    )
    var log_transition = _trace_safe_log(transition_tensor)
    var log_prior_theta = _trace_safe_log(q_static_state)
    var log_action_prior = _trace_safe_log(action_prior)
    var trace = List[Float32]()
    for iteration in range(n_iterations):
        var cursor = prefix_size + iteration * history_size
        var reduced = _trace_slice(diagnostic, cursor, reduced_size)
        cursor += reduced_size
        var log_fwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var log_bwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var q_u = _trace_slice(diagnostic, cursor, action_size)
        cursor += action_size
        var cavity = _trace_slice(diagnostic, cursor, cavity_size)
        cursor += cavity_size
        var dyn_regions = _trace_slice(diagnostic, cursor, dyn_region_size)
        trace.append(
            compute_dyn_channel_vfe(
                log_transition,
                reduced,
                log_fwd,
                log_bwd,
                q_u,
                cavity,
                log_prior_theta,
                log_action_prior,
                dyn_regions,
                horizon,
                n_states,
                n_static,
                n_actions,
            )
        )
    var result = _trace_slice(diagnostic, 0, prefix_size)
    for value in trace:
        result.append(value)
    return result^


def vbp_channel_planning_with_vfe(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    theta_goal: Bool,
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense JAX `vbp_channel_convergence`, flattened with trace last."""
    var no_indices = List[Int]()
    _validate_trace_goal_shape(goal, theta_goal, n_states, n_static)
    var diagnostic = _vbp_channel_planning(
        q_current_state,
        q_static_state,
        no_indices,
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
        theta_goal,
        True,
    )
    var action_channel_size = horizon * n_states * n_actions
    var prefix_size = n_actions + action_channel_size
    var reduced_size = horizon * n_states * n_states * n_actions
    var message_size = (horizon + 1) * n_states
    var action_size = horizon * n_actions
    var cavity_size = horizon * n_static
    var dyn_region_size = reduced_size * n_static
    var history_size = (
        reduced_size
        + 2 * message_size
        + action_size
        + cavity_size
        + dyn_region_size
    )
    var log_transition = _trace_safe_log(transition_tensor)
    var log_prior_theta = _trace_safe_log(q_static_state)
    var log_action_prior = _trace_safe_log(action_prior)
    var trace = List[Float32]()
    for iteration in range(n_iterations):
        var cursor = prefix_size + iteration * history_size
        var reduced = _trace_slice(diagnostic, cursor, reduced_size)
        cursor += reduced_size
        var log_fwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var log_bwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var q_u = _trace_slice(diagnostic, cursor, action_size)
        cursor += action_size
        var cavity = _trace_slice(diagnostic, cursor, cavity_size)
        cursor += cavity_size
        var dyn_regions = _trace_slice(diagnostic, cursor, dyn_region_size)
        trace.append(
            compute_vbp_channel_vfe(
                log_transition,
                reduced,
                log_fwd,
                log_bwd,
                q_u,
                cavity,
                log_prior_theta,
                log_action_prior,
                dyn_regions,
                horizon,
                n_states,
                n_static,
                n_actions,
            )
        )
    var result = _trace_slice(diagnostic, 0, prefix_size)
    for value in trace:
        result.append(value)
    return result^


def precise_info_seeking_planning_with_vfe(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    theta_goal: Bool,
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense JAX `precise_info_seeking_convergence`, trace last."""
    _validate_trace_goal_shape(goal, theta_goal, n_states, n_static)
    var diagnostic = _precise_info_seeking_planning(
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
        theta_goal,
        True,
        True,
    )
    var action_channel_size = horizon * n_states * n_actions
    var obs_channel_size = (
        (horizon + 1) * n_fov * n_obs_types * n_states * n_static
    )
    var prefix_size = n_actions + action_channel_size + obs_channel_size
    var reduced_size = horizon * n_states * n_states * n_actions
    var message_size = (horizon + 1) * n_states
    var action_size = horizon * n_actions
    var dyn_region_size = reduced_size * n_static
    var history_size = (
        reduced_size
        + 2 * message_size
        + action_size
        + dyn_region_size
        + obs_channel_size
    )
    var log_transition = _trace_safe_log(transition_tensor)
    var log_prior_theta = _trace_safe_log(q_static_state)
    var log_action_prior = _trace_safe_log(action_prior)
    var log_prior_dyn = _trace_tiled_prior(log_prior_theta, horizon)
    var trace = List[Float32]()
    for iteration in range(n_iterations):
        var cursor = prefix_size + iteration * history_size
        var reduced = _trace_slice(diagnostic, cursor, reduced_size)
        cursor += reduced_size
        var log_fwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var log_bwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var q_u = _trace_slice(diagnostic, cursor, action_size)
        cursor += action_size
        var dyn_regions = _trace_slice(diagnostic, cursor, dyn_region_size)
        cursor += dyn_region_size
        var obs_regions = _trace_slice(diagnostic, cursor, obs_channel_size)
        trace.append(
            compute_precise_info_seeking_vfe(
                log_transition,
                reduced,
                log_fwd,
                log_bwd,
                q_u,
                log_prior_dyn,
                log_prior_theta,
                log_action_prior,
                dyn_regions,
                obs_regions,
                horizon,
                n_states,
                n_static,
                n_actions,
                n_fov,
                n_obs_types,
            )
        )
    var result = _trace_slice(diagnostic, 0, prefix_size)
    for value in trace:
        result.append(value)
    return result^


def nuijten_mp_planning_with_vfe(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    theta_goal: Bool,
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense JAX `nuijten_mp_convergence`, flattened with trace last."""
    var no_indices = List[Int]()
    _validate_trace_goal_shape(goal, theta_goal, n_states, n_static)
    var diagnostic = _nuijten_mp_planning(
        q_current_state,
        q_static_state,
        no_indices,
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
        theta_goal,
        True,
    )
    var dyn_region_size = horizon * n_states * n_states * n_static * n_actions
    var obs_region_size = (
        (horizon + 1) * n_fov * n_obs_types * n_states * n_static
    )
    var prefix_size = n_actions + dyn_region_size + obs_region_size
    var message_size = (horizon + 1) * n_states
    var action_size = horizon * n_actions
    var cavity_dyn_size = horizon * n_static
    var cavity_obs_size = (horizon + 1) * n_static
    var history_size = (
        2 * message_size
        + action_size
        + dyn_region_size
        + obs_region_size
        + cavity_dyn_size
        + cavity_obs_size
        + action_size
    )
    var log_transition = _trace_safe_log(transition_tensor)
    var log_transition_tiled = _trace_transition_kernel_tiled(
        log_transition, horizon, n_states, n_static, n_actions
    )
    var log_observation = _trace_safe_log(observation_tensor)
    var log_prior_theta = _trace_safe_log(q_static_state)
    var trace = List[Float32]()
    for iteration in range(n_iterations):
        var cursor = prefix_size + iteration * history_size
        var log_fwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var log_bwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var q_u = _trace_slice(diagnostic, cursor, action_size)
        cursor += action_size
        var dyn_regions = _trace_slice(diagnostic, cursor, dyn_region_size)
        cursor += dyn_region_size
        var obs_regions = _trace_slice(diagnostic, cursor, obs_region_size)
        cursor += obs_region_size
        var cavity_dyn = _trace_slice(diagnostic, cursor, cavity_dyn_size)
        cursor += cavity_dyn_size
        var cavity_obs = _trace_slice(diagnostic, cursor, cavity_obs_size)
        cursor += cavity_obs_size
        var action_prior_per_t = _trace_slice(diagnostic, cursor, action_size)
        trace.append(
            compute_nuijten_vfe(
                dyn_regions,
                obs_regions,
                log_fwd,
                log_bwd,
                q_u,
                log_transition_tiled,
                log_observation,
                cavity_dyn,
                cavity_obs,
                log_prior_theta,
                action_prior_per_t,
                horizon,
                n_states,
                n_static,
                n_actions,
                n_fov,
                n_obs_types,
            )
        )
    var result = _trace_slice(diagnostic, 0, prefix_size)
    for value in trace:
        result.append(value)
    return result^


def active_inference_planning_with_vfe(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    theta_goal: Bool,
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_fov: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Dense JAX `active_inference_convergence`, trace last."""
    _validate_trace_goal_shape(goal, theta_goal, n_states, n_static)
    var log_transition = _trace_safe_log(transition_tensor)
    var log_prior_theta = _trace_safe_log(q_static_state)
    var log_base = _compute_dense_log_base(
        log_transition, log_prior_theta, n_states, n_actions, n_static
    )
    var diagnostic = _active_inference_planning(
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
        theta_goal,
        True,
        True,
    )
    var dyn_channel_size = horizon * n_states * n_states * n_actions
    var obs_channel_size = (
        (horizon + 1) * n_fov * n_obs_types * n_states * n_static
    )
    var prefix_size = n_actions + dyn_channel_size + obs_channel_size
    var reduced_size = dyn_channel_size
    var message_size = (horizon + 1) * n_states
    var action_size = horizon * n_actions
    var dyn_region_size = dyn_channel_size * n_static
    var history_size = (
        reduced_size
        + 2 * message_size
        + action_size
        + dyn_region_size
        + obs_channel_size
    )
    var log_action_prior = _trace_safe_log(action_prior)
    var log_prior_dyn = _trace_tiled_prior(log_prior_theta, horizon)
    var trace = List[Float32]()
    for iteration in range(n_iterations):
        var cursor = prefix_size + iteration * history_size
        var reduced = _trace_slice(diagnostic, cursor, reduced_size)
        cursor += reduced_size
        var log_fwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var log_bwd = _trace_slice(diagnostic, cursor, message_size)
        cursor += message_size
        var q_u = _trace_slice(diagnostic, cursor, action_size)
        cursor += action_size
        var dyn_regions = _trace_slice(diagnostic, cursor, dyn_region_size)
        cursor += dyn_region_size
        var obs_regions = _trace_slice(diagnostic, cursor, obs_channel_size)
        trace.append(
            compute_active_inference_vfe(
                log_transition,
                reduced,
                log_fwd,
                log_bwd,
                q_u,
                log_prior_dyn,
                log_prior_theta,
                log_action_prior,
                dyn_regions,
                obs_regions,
                horizon,
                n_states,
                n_static,
                n_actions,
                n_fov,
                n_obs_types,
            )
        )
    var result = _trace_slice(diagnostic, 0, prefix_size)
    for value in trace:
        result.append(value)
    return result^
