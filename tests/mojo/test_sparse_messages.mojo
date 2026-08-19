from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.numerics import LOG_ZERO, safe_log
from aif_mojo.sparse_messages import (
    compute_log_base_sparse,
    sparse_dyn_to_theta,
    sparse_dyn_to_theta_weighted,
    sparse_dyn_to_theta_dyn_channel,
    sparse_dyn_channels_and_pair,
    sparse_dyn_channels_and_pair_dyn_channel,
    sparse_dyn_channels_and_pair_weighted,
    sparse_efe_action_prior,
    sparse_pair_marginal,
    sparse_pair_marginal_weighted,
    sparse_reduced,
    sparse_reduced_weighted,
    sparse_reduced_dyn_channel,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def transition_indices() -> List[Int]:
    # Shape (old=2, action=1, static=2): static 0 preserves, static 1 -> state 1.
    var result = List[Int]()
    result.append(0)
    result.append(1)
    result.append(1)
    result.append(1)
    return result^


def log_pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(safe_log(a))
    result.append(safe_log(b))
    return result^


def probability_pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def test_compute_log_base_sparse_accumulates_static_paths() raises:
    var result = compute_log_base_sparse(
        transition_indices(), log_pair(0.25, 0.75), 2, 1, 2
    )
    # Shape (old, new, action).
    assert_close(exp(result[0]), 0.25)
    assert_close(exp(result[1]), 0.75)
    assert_equal(result[2], LOG_ZERO)
    assert_close(exp(result[3]), 1.0)


def test_sparse_reduced_returns_new_old_axis_order() raises:
    var cavities = List[Float32]()
    cavities.extend(log_pair(0.25, 0.75))
    cavities.extend(log_pair(0.6, 0.4))
    var result = sparse_reduced(transition_indices(), cavities, 2, 2, 1, 2)
    # Shape (time, new, old, action).
    assert_close(exp(result[0]), 0.25)
    assert_equal(result[1], LOG_ZERO)
    assert_close(exp(result[2]), 0.75)
    assert_close(exp(result[3]), 1.0)
    assert_close(exp(result[4]), 0.6)
    assert_equal(result[5], LOG_ZERO)
    assert_close(exp(result[6]), 0.4)
    assert_close(exp(result[7]), 1.0)


def test_sparse_reduced_weighted_adds_old_state_weights() raises:
    var cavities = log_pair(0.25, 0.75)
    var weights = log_pair(0.5, 0.2)
    var result = sparse_reduced_weighted(
        transition_indices(), cavities, weights, 1, 2, 1, 2
    )
    assert_close(exp(result[0]), 0.125)
    assert_equal(result[1], LOG_ZERO)
    assert_close(exp(result[2]), 0.375)
    assert_close(exp(result[3]), 0.2)


def test_sparse_dyn_to_theta_matches_deterministic_paths() raises:
    var log_fwd = List[Float32]()
    log_fwd.extend(log_pair(0.6, 0.4))
    log_fwd.extend(log_pair(0.5, 0.5))
    var log_bwd = List[Float32]()
    log_bwd.extend(log_pair(0.5, 0.5))
    log_bwd.extend(log_pair(0.7, 0.3))
    var local = List[Float32]()
    for _ in range(4):
        local.append(0.0)
    var action = List[Float32]()
    action.append(0.0)

    var result = sparse_dyn_to_theta(
        transition_indices(), log_fwd, log_bwd, local, action, 1, 2, 1, 2
    )
    assert_close(exp(result[0]), 0.54)
    assert_close(exp(result[1]), 0.3)

    var weights = log_pair(0.5, 0.2)
    var weighted = sparse_dyn_to_theta_weighted(
        transition_indices(),
        log_fwd,
        log_bwd,
        local,
        action,
        weights,
        1,
        2,
        1,
        2,
    )
    assert_close(exp(weighted[0]), 0.234)
    assert_close(exp(weighted[1]), 0.114)


def test_sparse_pair_marginal_accumulates_static_beliefs() raises:
    var log_fwd = List[Float32]()
    log_fwd.extend(log_pair(0.6, 0.4))
    log_fwd.extend(log_pair(0.5, 0.5))
    var log_bwd = List[Float32]()
    log_bwd.extend(log_pair(0.5, 0.5))
    log_bwd.extend(log_pair(0.7, 0.3))
    var local = List[Float32]()
    for _ in range(4):
        local.append(0.0)
    var action = List[Float32]()
    action.append(0.0)
    var cavity = log_pair(0.25, 0.75)

    var result = sparse_pair_marginal(
        transition_indices(),
        log_fwd,
        log_bwd,
        local,
        cavity,
        action,
        1,
        2,
        1,
        2,
    )
    assert_close(exp(result[0]), 0.24)
    assert_close(exp(result[1]), 0.12)

    var weighted = sparse_pair_marginal_weighted(
        transition_indices(),
        log_fwd,
        log_bwd,
        local,
        cavity,
        action,
        log_pair(0.5, 0.2),
        1,
        2,
        1,
        2,
    )
    assert_close(exp(weighted[0]), 0.12)
    assert_close(exp(weighted[1]), 0.024)


def test_sparse_dyn_channel_reduction_applies_inverse_channel_weight() raises:
    var cavity = log_pair(0.25, 0.75)
    # Shape (time=1, old=2, new=2, action=1).
    var channels = List[Float32]()
    channels.extend(log_pair(0.4, 0.6))
    channels.extend(log_pair(0.2, 0.8))
    var action_channel = List[Float32]()
    action_channel.append(0.0)
    action_channel.append(0.0)
    var result = sparse_reduced_dyn_channel(
        transition_indices(), cavity, channels, action_channel, 1, 2, 1, 2
    )
    assert_close(exp(result[0]), 0.625)
    assert_equal(result[1], LOG_ZERO)
    assert_close(exp(result[2]), 1.25)
    assert_close(exp(result[3]), 1.25)


def test_sparse_dyn_channel_theta_and_channel_updates_match_jax() raises:
    var log_fwd = List[Float32]()
    log_fwd.extend(log_pair(0.6, 0.4))
    log_fwd.extend(log_pair(0.5, 0.5))
    var log_bwd = List[Float32]()
    log_bwd.extend(log_pair(0.5, 0.5))
    log_bwd.extend(log_pair(0.7, 0.3))
    var local = List[Float32]()
    for _ in range(4):
        local.append(0.0)
    var action = List[Float32]()
    action.append(0.0)
    var channels = List[Float32]()
    channels.extend(log_pair(0.4, 0.6))
    channels.extend(log_pair(0.2, 0.8))
    var action_channel = List[Float32]()
    action_channel.append(0.0)
    action_channel.append(0.0)

    var theta = sparse_dyn_to_theta_dyn_channel(
        transition_indices(),
        log_fwd,
        log_bwd,
        local,
        action,
        channels,
        action_channel,
        1,
        2,
        1,
        2,
    )
    assert_close(exp(theta[0]), 1.2)
    assert_close(exp(theta[1]), 0.45)

    var result = sparse_dyn_channels_and_pair_dyn_channel(
        transition_indices(),
        log_fwd,
        log_bwd,
        local,
        log_pair(0.25, 0.75),
        action,
        channels,
        action_channel,
        1,
        2,
        1,
        2,
    )
    # First T*S*S*A values are channels in (time, old, new, action) order.
    assert_close(exp(result[0]), 0.53846157)
    assert_close(exp(result[1]), 0.46153846)
    assert_equal(result[2], LOG_ZERO)
    assert_close(exp(result[3]), 1.0)
    # Remaining T*S*A values are the pair marginal.
    assert_close(exp(result[4]), 0.4875)
    assert_close(exp(result[5]), 0.15)


def test_sparse_channel_and_pair_updates_support_optional_weights() raises:
    var log_fwd = List[Float32]()
    log_fwd.extend(log_pair(0.6, 0.4))
    log_fwd.extend(log_pair(0.5, 0.5))
    var log_bwd = List[Float32]()
    log_bwd.extend(log_pair(0.5, 0.5))
    log_bwd.extend(log_pair(0.7, 0.3))
    var local = List[Float32]()
    for _ in range(4):
        local.append(0.0)
    var action = List[Float32]()
    action.append(0.0)
    var cavity = log_pair(0.25, 0.75)

    var result = sparse_dyn_channels_and_pair(
        transition_indices(),
        log_fwd,
        log_bwd,
        local,
        cavity,
        action,
        1,
        2,
        1,
        2,
    )
    assert_close(exp(result[0]), 0.4375)
    assert_close(exp(result[1]), 0.5625)
    assert_equal(result[2], LOG_ZERO)
    assert_close(exp(result[3]), 1.0)
    assert_close(exp(result[4]), 0.24)
    assert_close(exp(result[5]), 0.12)

    var weighted = sparse_dyn_channels_and_pair_weighted(
        transition_indices(),
        log_fwd,
        log_bwd,
        local,
        cavity,
        action,
        log_pair(0.5, 0.2),
        1,
        2,
        1,
        2,
    )
    assert_close(exp(weighted[0]), 0.4375)
    assert_close(exp(weighted[1]), 0.5625)
    assert_equal(weighted[2], LOG_ZERO)
    assert_close(exp(weighted[3]), 1.0)
    assert_close(exp(weighted[4]), 0.12)
    assert_close(exp(weighted[5]), 0.024)


def test_sparse_efe_action_prior_is_normalized_and_masks_actions() raises:
    # Shape (old=2, action=2, static=2).
    var transitions = List[Int]()
    transitions.append(0)
    transitions.append(1)
    transitions.append(1)
    transitions.append(0)
    transitions.append(1)
    transitions.append(0)
    transitions.append(0)
    transitions.append(1)
    var log_fwd = List[Float32]()
    log_fwd.extend(log_pair(0.6, 0.4))
    log_fwd.extend(log_pair(0.5, 0.5))
    var log_bwd = List[Float32]()
    log_bwd.extend(log_pair(0.5, 0.5))
    log_bwd.extend(log_pair(0.7, 0.3))
    var local = List[Float32]()
    for _ in range(4):
        local.append(0.0)
    var action = log_pair(0.5, 0.5)
    var cavity = log_pair(0.7, 0.3)

    var unmasked = sparse_efe_action_prior(
        transitions,
        log_fwd,
        log_bwd,
        local,
        cavity,
        action,
        probability_pair(1.0, 1.0),
        1,
        2,
        2,
        2,
    )
    assert_close(unmasked[0], 0.5)
    assert_close(unmasked[1], 0.5)

    var masked = sparse_efe_action_prior(
        transitions,
        log_fwd,
        log_bwd,
        local,
        cavity,
        action,
        probability_pair(1.0, 0.0),
        1,
        2,
        2,
        2,
    )
    assert_close(masked[0], 1.0)
    assert_true(masked[1] < 1.0e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
