from std.collections import List
from std.math import exp, log

from aif_mojo.numerics import (
    EPSILON,
    LOG_ZERO,
    logaddexp,
    logsumexp,
    safe_log_div,
    softmax,
)


def _log_zero_buffer(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(LOG_ZERO)
    return result^


def compute_log_base_sparse(
    transition_indices: List[Int],
    log_weights: List[Float32],
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Marginalize static configurations without a dense transition tensor."""
    debug_assert(
        len(transition_indices) == n_states * n_actions * n_static,
        "transition index shape mismatch",
    )
    debug_assert(len(log_weights) == n_static, "static weight shape mismatch")

    var result = _log_zero_buffer(n_states * n_states * n_actions)
    for old_idx in range(n_states):
        for action_idx in range(n_actions):
            for static_idx in range(n_static):
                var transition_offset = (
                    old_idx * n_actions + action_idx
                ) * n_static + static_idx
                var new_idx = transition_indices[transition_offset]
                debug_assert(
                    new_idx >= 0 and new_idx < n_states, "invalid next state"
                )
                var result_offset = (
                    old_idx * n_states + new_idx
                ) * n_actions + action_idx
                result[result_offset] = logaddexp(
                    result[result_offset], log_weights[static_idx]
                )
    return result^


def sparse_reduced(
    transition_indices: List[Int],
    log_cavity: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Return per-time reductions in (time, new, old, action) order."""
    debug_assert(
        len(transition_indices) == n_states * n_actions * n_static,
        "transition index shape mismatch",
    )
    debug_assert(len(log_cavity) == horizon * n_static, "cavity shape mismatch")

    var result = _log_zero_buffer(horizon * n_states * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                for static_idx in range(n_static):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    debug_assert(
                        new_idx >= 0 and new_idx < n_states,
                        "invalid next state",
                    )
                    var result_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    var contribution = log_cavity[
                        time_idx * n_static + static_idx
                    ]
                    result[result_offset] = logaddexp(
                        result[result_offset], contribution
                    )
    return result^


def sparse_reduced_weighted(
    transition_indices: List[Int],
    log_cavity: List[Float32],
    log_weight: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Sparse reduction with additive (time, old, action) log weights."""
    debug_assert(
        len(log_weight) == horizon * n_states * n_actions,
        "weight shape mismatch",
    )
    debug_assert(
        len(transition_indices) == n_states * n_actions * n_static,
        "transition index shape mismatch",
    )
    debug_assert(len(log_cavity) == horizon * n_static, "cavity shape mismatch")

    var result = _log_zero_buffer(horizon * n_states * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var weight_offset = (
                    time_idx * n_states + old_idx
                ) * n_actions + action_idx
                for static_idx in range(n_static):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    debug_assert(
                        new_idx >= 0 and new_idx < n_states,
                        "invalid next state",
                    )
                    var result_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    var contribution = (
                        log_cavity[time_idx * n_static + static_idx]
                        + log_weight[weight_offset]
                    )
                    result[result_offset] = logaddexp(
                        result[result_offset], contribution
                    )
    return result^


def sparse_reduced_dyn_channel(
    transition_indices: List[Int],
    log_cavity: List[Float32],
    log_dyn_channels: List[Float32],
    log_action_channel: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Sparse reduction for T * r(action|old) / r(new|old,action)."""
    debug_assert(
        len(log_dyn_channels) == horizon * n_states * n_states * n_actions,
        "dynamic channel shape mismatch",
    )
    debug_assert(
        len(log_action_channel) == horizon * n_states * n_actions,
        "action channel shape mismatch",
    )
    var result = _log_zero_buffer(horizon * n_states * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var action_offset = (
                    time_idx * n_states + old_idx
                ) * n_actions + action_idx
                for static_idx in range(n_static):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var channel_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var result_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    var contribution = (
                        log_cavity[time_idx * n_static + static_idx]
                        + safe_log_div(0.0, log_dyn_channels[channel_offset])
                        + log_action_channel[action_offset]
                    )
                    result[result_offset] = logaddexp(
                        result[result_offset], contribution
                    )
    return result^


def sparse_dyn_to_theta_dyn_channel(
    transition_indices: List[Int],
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
    """Dynamics-to-theta messages for theta-dependent channel weights."""
    var result = List[Float32]()
    for time_idx in range(horizon):
        for static_idx in range(n_static):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var channel_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var action_channel_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action[action_idx]
                        + safe_log_div(0.0, log_dyn_channels[channel_offset])
                        + log_action_channel[action_channel_offset]
                    )
            result.append(logsumexp(terms))
    return result^


def _channels_and_pair_from_theta_marginal(
    log_theta_marginal: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Concatenate normalized channels with their pair marginal."""
    var channels = List[Float32]()
    var pair = List[Float32]()
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var terms = List[Float32]()
                for new_idx in range(n_states):
                    var offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    terms.append(log_theta_marginal[offset])
                pair.append(logsumexp(terms))

    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var channel_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var pair_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    channels.append(
                        log_theta_marginal[channel_offset] - pair[pair_offset]
                    )

    for value in pair:
        channels.append(value)
    return channels^


def sparse_dyn_channels_and_pair_dyn_channel(
    transition_indices: List[Int],
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
    """Return concatenated dynamic channels and pair marginal."""
    var marginal = _log_zero_buffer(horizon * n_states * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                for static_idx in range(n_static):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var marginal_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var action_channel_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    var contribution = (
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action[action_idx]
                        + log_cavity_dyn[time_idx * n_static + static_idx]
                        + safe_log_div(0.0, log_dyn_channels[marginal_offset])
                        + log_action_channel[action_channel_offset]
                    )
                    marginal[marginal_offset] = logaddexp(
                        marginal[marginal_offset], contribution
                    )
    return _channels_and_pair_from_theta_marginal(
        marginal, horizon, n_states, n_actions
    )


def _sparse_dyn_channels_and_pair(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    log_kernel_weight: List[Float32],
    use_weight: Bool,
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    var marginal = _log_zero_buffer(horizon * n_states * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var weight_offset = (
                    time_idx * n_states + old_idx
                ) * n_actions + action_idx
                for static_idx in range(n_static):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var marginal_offset = (
                        (time_idx * n_states + old_idx) * n_states + new_idx
                    ) * n_actions + action_idx
                    var contribution = (
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action[action_idx]
                        + log_cavity_dyn[time_idx * n_static + static_idx]
                    )
                    if use_weight:
                        contribution += log_kernel_weight[weight_offset]
                    marginal[marginal_offset] = logaddexp(
                        marginal[marginal_offset], contribution
                    )
    return _channels_and_pair_from_theta_marginal(
        marginal, horizon, n_states, n_actions
    )


def sparse_dyn_channels_and_pair(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Return unweighted sparse channels followed by the pair marginal."""
    var no_weights = List[Float32]()
    return _sparse_dyn_channels_and_pair(
        transition_indices,
        log_fwd,
        log_bwd,
        log_local_to_x,
        log_cavity_dyn,
        log_action,
        no_weights,
        False,
        horizon,
        n_states,
        n_actions,
        n_static,
    )


def sparse_dyn_channels_and_pair_weighted(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    log_kernel_weight: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Return weighted sparse channels followed by the pair marginal."""
    return _sparse_dyn_channels_and_pair(
        transition_indices,
        log_fwd,
        log_bwd,
        log_local_to_x,
        log_cavity_dyn,
        log_action,
        log_kernel_weight,
        True,
        horizon,
        n_states,
        n_actions,
        n_static,
    )


def sparse_efe_action_prior(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action_per_t: List[Float32],
    action_mask: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Compute the sparse deterministic EFE action prior in probability space.
    """
    var log_normalizers = List[Float32]()
    for time_idx in range(horizon):
        for action_idx in range(n_actions):
            var terms = List[Float32]()
            for static_idx in range(n_static):
                for old_idx in range(n_states):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    terms.append(
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_cavity_dyn[time_idx * n_static + static_idx]
                        + log_action_per_t[time_idx * n_actions + action_idx]
                    )
            log_normalizers.append(logsumexp(terms))

    var efe = List[Float32]()
    for time_idx in range(horizon):
        for action_idx in range(n_actions):
            var value = Float32(0.0)
            var normalizer = log_normalizers[time_idx * n_actions + action_idx]
            for static_idx in range(n_static):
                for old_idx in range(n_states):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var logit = (
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_cavity_dyn[time_idx * n_static + static_idx]
                        + log_action_per_t[time_idx * n_actions + action_idx]
                    )
                    var probability = exp(logit - normalizer)
                    var conditional = probability / (probability + EPSILON)
                    value -= probability * log(conditional + EPSILON)
            efe.append(value)

    var result = List[Float32]()
    for time_idx in range(horizon):
        var masked_logits = List[Float32]()
        for action_idx in range(n_actions):
            if action_mask[action_idx] > 0.0:
                masked_logits.append(efe[time_idx * n_actions + action_idx])
            else:
                masked_logits.append(LOG_ZERO)
        var probabilities = softmax(masked_logits)
        for probability in probabilities:
            result.append(probability)
    return result^


def sparse_dyn_to_theta(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_action_per_t: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Compute sparse dynamics-to-static messages in (time, static) order."""
    debug_assert(len(log_fwd) == (horizon + 1) * n_states, "fwd shape mismatch")
    debug_assert(len(log_bwd) == (horizon + 1) * n_states, "bwd shape mismatch")
    debug_assert(
        len(log_local_to_x) == (horizon + 1) * n_states,
        "local shape mismatch",
    )
    debug_assert(
        len(log_action_per_t) == horizon * n_actions,
        "action shape mismatch",
    )

    var result = List[Float32]()
    for time_idx in range(horizon):
        for static_idx in range(n_static):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    terms.append(
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action_per_t[time_idx * n_actions + action_idx]
                    )
            result.append(logsumexp(terms))
    return result^


def sparse_dyn_to_theta_weighted(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_action_per_t: List[Float32],
    log_kernel_weight: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dynamics-to-static messages with (time, old, action) kernel weights."""
    debug_assert(
        len(log_kernel_weight) == horizon * n_states * n_actions,
        "kernel weight shape mismatch",
    )
    var result = List[Float32]()
    for time_idx in range(horizon):
        for static_idx in range(n_static):
            var terms = List[Float32]()
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var weight_offset = (
                        time_idx * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action_per_t[time_idx * n_actions + action_idx]
                        + log_kernel_weight[weight_offset]
                    )
            result.append(logsumexp(terms))
    return result^


def sparse_pair_marginal(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Compute log q(time, old, action) by marginalizing static state."""
    var result = _log_zero_buffer(horizon * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var result_offset = (
                    time_idx * n_states + old_idx
                ) * n_actions + action_idx
                for static_idx in range(n_static):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var contribution = (
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action[action_idx]
                        + log_cavity_dyn[time_idx * n_static + static_idx]
                    )
                    result[result_offset] = logaddexp(
                        result[result_offset], contribution
                    )
    return result^


def sparse_pair_marginal_weighted(
    transition_indices: List[Int],
    log_fwd: List[Float32],
    log_bwd: List[Float32],
    log_local_to_x: List[Float32],
    log_cavity_dyn: List[Float32],
    log_action: List[Float32],
    log_kernel_weight: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Sparse pair marginal with additive (time, old, action) weights."""
    var result = _log_zero_buffer(horizon * n_states * n_actions)
    for time_idx in range(horizon):
        for old_idx in range(n_states):
            for action_idx in range(n_actions):
                var result_offset = (
                    time_idx * n_states + old_idx
                ) * n_actions + action_idx
                for static_idx in range(n_static):
                    var transition_offset = (
                        old_idx * n_actions + action_idx
                    ) * n_static + static_idx
                    var new_idx = transition_indices[transition_offset]
                    var contribution = (
                        log_fwd[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_bwd[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action[action_idx]
                        + log_cavity_dyn[time_idx * n_static + static_idx]
                        + log_kernel_weight[result_offset]
                    )
                    result[result_offset] = logaddexp(
                        result[result_offset], contribution
                    )
    return result^
