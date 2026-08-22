from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_true

from aif_mojo.active_inference import (
    PreparedDenseAIFMP,
    active_inference_planning_dense,
    active_inference_planning_dense_theta_goal,
    active_inference_planning_dense_until_converged,
    active_inference_planning_dense_specialized,
    active_inference_planning_precomputed,
    active_inference_planning_precomputed_theta_goal,
    active_inference_planning_precomputed_with_preferences,
    compute_dyn_kernels_aif,
    precompute_obs_channels,
    precompute_pref_to_x,
)
from aif_mojo.numerics import LOG_ZERO, safe_log
from aif_mojo.sparse_messages import compute_log_base_sparse


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
    return result^


def transition() -> List[Float32]:
    # Shape (new state, old state, theta, action).
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.4, 0.7))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.6, 0.1))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.6, 0.3))
    result.extend(pair(0.7, 0.2))
    result.extend(pair(0.4, 0.9))
    return result^


def observation() -> List[Float32]:
    # Shape (field of view, observation, state, theta).
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def theta_goal() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.15, 0.75))
    result.extend(pair(0.85, 0.25))
    return result^


def dense_terminal(action_prior: List[Float32]) -> List[Float32]:
    return active_inference_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        pair(0.15, 0.85),
        action_prior,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def precomputed_local(theta_preference: Bool) -> List[Float32]:
    var log_prior = log_values(pair(0.55, 0.45))
    var local = precompute_obs_channels(
        log_values(observation()), log_prior, 2, 0.5, 2, 2, 1, 2
    )
    if theta_preference:
        var preference = precompute_pref_to_x(
            log_values(theta_goal()), log_prior, 2, 2
        )
        for index in range(2):
            local[index] += preference[index]
    return local^


def dense_log_base() -> List[Float32]:
    # JAX reference values for theta-marginalizing transition() with [0.55, 0.45].
    var result = List[Float32]()
    result.extend(pair(-0.3930426, -0.8556661))
    result.extend(pair(-1.1239302, -0.55338514))
    result.extend(pair(-0.8324092, -0.7236063))
    result.extend(pair(-0.5709296, -0.66358846))
    return result^


def test_active_inference_dynamics_kernel_matches_complete_jax_fixture() raises:
    # Transition kernel shape (old, new, theta, action).
    var transition_kernel = List[Float32]()
    transition_kernel.extend(pair(0.8, 0.0))
    transition_kernel.extend(pair(0.4, 0.6))
    transition_kernel.extend(pair(0.2, 1.0))
    transition_kernel.extend(pair(0.6, 0.4))
    transition_kernel.extend(pair(0.3, 0.7))
    transition_kernel.extend(pair(0.0, 0.5))
    transition_kernel.extend(pair(0.7, 0.3))
    transition_kernel.extend(pair(1.0, 0.5))
    var action_channels = List[Float32]()
    action_channels.extend(pair(0.75, 0.25))
    action_channels.extend(pair(0.4, 0.6))
    action_channels.extend(pair(0.2, 0.8))
    action_channels.extend(pair(0.9, 0.1))
    var dyn_channels = List[Float32]()
    dyn_channels.extend(pair(0.6, 0.0))
    dyn_channels.extend(pair(0.4, 1.0))
    dyn_channels.extend(pair(0.3, 0.7))
    dyn_channels.extend(pair(0.7, 0.3))
    dyn_channels.extend(pair(0.1, 0.8))
    dyn_channels.extend(pair(0.9, 0.2))
    dyn_channels.extend(pair(0.0, 0.4))
    dyn_channels.extend(pair(1.0, 0.6))
    var actual = compute_dyn_kernels_aif(
        log_values(transition_kernel),
        log_values(action_channels),
        log_values(dyn_channels),
        2,
        2,
        2,
        2,
    )
    var expected = List[Float32]()
    expected.append(0.0)
    expected.append(LOG_ZERO)
    expected.append(-0.69314718)
    expected.append(LOG_ZERO)
    expected.append(-0.98082936)
    expected.append(-1.38629436)
    expected.append(0.11778304)
    expected.append(-2.30258512)
    expected.append(-0.91629070)
    expected.append(-0.51082557)
    expected.append(LOG_ZERO)
    expected.append(-0.84729779)
    expected.append(-0.91629070)
    expected.append(-0.51082557)
    expected.append(-0.55961573)
    expected.append(0.0)
    expected.append(0.47000360)
    expected.append(LOG_ZERO)
    expected.append(-0.22314358)
    expected.append(-0.51082557)
    expected.append(-3.11351538)
    expected.append(1.38629436)
    expected.append(-2.01490307)
    expected.append(0.47000372)
    expected.append(LOG_ZERO)
    expected.append(-1.74296939)
    expected.append(LOG_ZERO)
    expected.append(-2.07944155)
    expected.append(-0.46203551)
    expected.append(-2.99573231)
    expected.append(-0.10536055)
    expected.append(-2.48490667)
    assert_true(len(actual) == 32)
    for index in range(32):
        assert_close(actual[index], expected[index])


def test_precomputed_local_messages_match_jax() raises:
    var log_prior = log_values(pair(0.55, 0.45))
    var obs_to_x = precompute_obs_channels(
        log_values(observation()), log_prior, 2, 0.5, 2, 2, 1, 2
    )
    assert_close(obs_to_x[0], -0.5310862)
    assert_close(obs_to_x[1], -0.88664943)

    var pref_to_x = precompute_pref_to_x(
        log_values(theta_goal()), log_prior, 2, 2
    )
    assert_close(pref_to_x[0], -0.8675006)
    assert_close(pref_to_x[1], -0.54472715)


def test_dense_terminal_matches_jax_and_returns_conditional_channels() raises:
    var result = dense_terminal(pair(0.6, 0.4))
    # action + dyn channels + observation channels
    assert_true(len(result) == 42)
    assert_close(result[0], 0.64464676)
    assert_close(result[1], 0.35535327)
    assert_close(result[2], -0.49224865)
    assert_close(result[17], -0.38458693)
    assert_close(result[18], -0.06756467)
    assert_close(result[41], -1.87442505)

    for time_idx in range(2):
        for old_idx in range(2):
            for action_idx in range(2):
                var zero = 2 + ((time_idx * 2 + old_idx) * 2 * 2 + action_idx)
                var one = zero + 2
                assert_close(exp(result[zero]) + exp(result[one]), 1.0)
    for time_idx in range(3):
        for state_idx in range(2):
            for static_idx in range(2):
                var zero = 18 + time_idx * 8 + state_idx * 2 + static_idx
                var one = zero + 4
                assert_close(exp(result[zero]) + exp(result[one]), 1.0)


def test_dense_theta_goal_matches_jax() raises:
    var result = active_inference_planning_dense_theta_goal(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        theta_goal(),
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 0.6366111)
    assert_close(result[1], 0.3633889)
    assert_close(result[2], -0.5406195)
    assert_close(result[17], -0.6678637)
    assert_close(result[18], -0.06756467)
    assert_close(result[41], -1.87442505)


def test_precomputed_terminal_and_theta_paths_match_jax() raises:
    var terminal = active_inference_planning_precomputed(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        dense_log_base(),
        precomputed_local(False),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    assert_true(len(terminal) == 18)
    assert_close(terminal[0], 0.65320015)
    assert_close(terminal[1], 0.34679985)
    assert_close(terminal[2], -0.47603077)
    assert_close(terminal[17], -0.39674857)

    var theta = active_inference_planning_precomputed_theta_goal(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        dense_log_base(),
        precomputed_local(True),
        theta_goal(),
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    assert_close(theta[0], 0.64426464)
    assert_close(theta[1], 0.35573533)
    assert_close(theta[2], -0.52189064)
    assert_close(theta[17], -0.6863634)


def test_sparse_precomputed_base_and_action_mask() raises:
    var indices = List[Int]()
    indices.append(0)
    indices.append(1)
    indices.append(1)
    indices.append(0)
    indices.append(1)
    indices.append(0)
    indices.append(0)
    indices.append(1)
    var sparse_base = compute_log_base_sparse(
        indices, log_values(pair(0.55, 0.45)), 2, 2, 2
    )
    var result = active_inference_planning_precomputed(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        sparse_base,
        precomputed_local(False),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    assert_close(result[0], 0.6509952)
    assert_close(result[1], 0.34900486)
    assert_close(result[2], -0.5683889)
    assert_close(result[17], -0.44133407)

    var masked = active_inference_planning_precomputed(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        sparse_base,
        precomputed_local(False),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    assert_close(masked[0], 1.0)
    assert_true(masked[1] < 1.0e-6)


def test_dense_early_stop_matches_fixed_iteration_result() raises:
    var fixed = active_inference_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    var stopped = active_inference_planning_dense_until_converged(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        20,
        0.5,
        1000.0,
        2,
        2,
        2,
        2,
        1,
        2,
        False,
    )
    assert_true(len(fixed) == 42)
    assert_true(len(stopped) == 48)
    for index in range(42):
        assert_close(stopped[index], fixed[index])
    assert_close(stopped[42], 1.0)
    assert_close(stopped[43], 2.0)
    assert_true(stopped[44] >= 0.0)
    assert_close(stopped[45], 0.5)
    assert_true(stopped[46] >= 0.0)
    assert_true(stopped[47] >= 0.0)


def test_prepared_and_compile_time_specialized_aif_match_public_api() raises:
    var expected = dense_terminal(pair(0.6, 0.4))
    var prepared = PreparedDenseAIFMP[2, 2](
        transition(),
        observation(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    var prepared_result = prepared.plan(pair(0.65, 0.35), pair(0.55, 0.45))
    var specialized = active_inference_planning_dense_specialized[2, 2](
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        transition(),
        observation(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    assert_true(len(prepared_result) == len(expected))
    assert_true(len(specialized) == len(expected))
    for index in range(len(expected)):
        assert_close(prepared_result[index], expected[index])
        assert_close(specialized[index], expected[index])


def test_precomputed_aif_accepts_time_indexed_preference_messages() raises:
    var preference = List[Float32]()
    preference.extend(pair(0.0, 0.0))
    preference.extend(pair(0.0, 0.0))
    preference.extend(log_values(pair(0.1, 0.9)))
    var result = active_inference_planning_precomputed_with_preferences(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        dense_log_base(),
        precomputed_local(False),
        preference,
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    assert_true(len(result) == 18)
    assert_close(result[0] + result[1], 1.0)
    assert_true(result[0] != result[1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
