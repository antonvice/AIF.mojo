from std.collections import List
from std.testing import TestSuite, assert_true

from aif_mojo.convergence import (
    action_cond_entropy,
    active_inference_planning_with_vfe,
    compute_active_inference_vfe,
    compute_bethe_vfe_loopy,
    compute_dyn_channel_vfe,
    compute_nuijten_vfe,
    compute_precise_info_seeking_vfe,
    compute_region_extended_vfe,
    compute_vbp_channel_vfe,
    dyn_channel_loopy_bp_planning_with_vfe,
    dyn_cond_entropy,
    energy,
    entropy,
    entropy_unnormalized,
    obs_cond_entropy,
    loopy_bp_planning_with_vfe,
    nuijten_mp_planning_with_vfe,
    precise_info_seeking_planning_with_vfe,
    region_extended_loopy_bp_planning_with_vfe,
    vbp_channel_planning_with_vfe,
)
from aif_mojo.numerics import LOG_ZERO, safe_log


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
    # Flat (new, old, theta, action).
    var result = List[Float32]()
    result.append(0.9)
    result.append(0.2)
    result.append(0.7)
    result.append(0.4)
    result.append(0.3)
    result.append(0.8)
    result.append(0.6)
    result.append(0.1)
    result.append(0.1)
    result.append(0.8)
    result.append(0.3)
    result.append(0.6)
    result.append(0.7)
    result.append(0.2)
    result.append(0.4)
    result.append(0.9)
    return result^


def reduced_placeholder() -> List[Float32]:
    var result = List[Float32]()
    for _ in range(16):
        result.append(0.0)
    return result^


def fwd_messages() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.8, 0.2))
    result.extend(pair(0.55, 0.45))
    result.extend(pair(0.4, 0.6))
    return log_values(result^)


def bwd_messages() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.45, 0.55))
    result.extend(pair(0.35, 0.65))
    result.extend(pair(0.2, 0.8))
    return log_values(result^)


def actions() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.7, 0.3))
    result.extend(pair(0.4, 0.6))
    return result^


def dyn_cavities() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.6, 0.4))
    result.extend(pair(0.3, 0.7))
    return log_values(result^)


def obs_cavities() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.5, 0.5))
    result.extend(pair(0.8, 0.2))
    result.extend(pair(0.25, 0.75))
    return log_values(result^)


def dyn_regions() -> List[Float32]:
    # Two independently normalized (old, new, theta, action) regions.
    var probabilities = List[Float32]()
    var denominator = Float32(136.0)
    for value in range(1, 17):
        probabilities.append(Float32(value) / denominator)
    denominator = 392.0
    for value in range(17, 33):
        probabilities.append(Float32(value) / denominator)
    return log_values(probabilities^)


def obs_region_probabilities() -> List[Float32]:
    # Three independently normalized (fov, observation, state, theta) regions.
    var result = List[Float32]()
    var denominators = List[Float32]()
    denominators.append(36.0)
    denominators.append(100.0)
    denominators.append(164.0)
    for time_idx in range(3):
        for offset in range(8):
            result.append(
                Float32(time_idx * 8 + offset + 1) / denominators[time_idx]
            )
    return result^


def observation_factor() -> List[Float32]:
    # Flat (fov, observation, state, theta).
    var result = List[Float32]()
    result.append(0.8)
    result.append(0.3)
    result.append(0.6)
    result.append(0.4)
    result.append(0.2)
    result.append(0.7)
    result.append(0.4)
    result.append(0.6)
    return log_values(result^)


def tiled_transition_kernel() -> List[Float32]:
    # Convert (new, old, theta, action) to tiled (time, old, new, theta, action).
    var log_transition = log_values(transition())
    var result = List[Float32]()
    for _ in range(2):
        for old_idx in range(2):
            for new_idx in range(2):
                for static_idx in range(2):
                    for action_idx in range(2):
                        var offset = (
                            (new_idx * 2 + old_idx) * 2 + static_idx
                        ) * 2 + action_idx
                        result.append(log_transition[offset])
    return result^


def action_prior_per_t() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.55, 0.45))
    result.extend(pair(0.2, 0.8))
    return result^


def bethe_fixture() -> Float32:
    return compute_bethe_vfe_loopy(
        log_values(transition()),
        reduced_placeholder(),
        fwd_messages(),
        bwd_messages(),
        actions(),
        dyn_cavities(),
        log_values(pair(0.5, 0.5)),
        log_values(pair(0.6, 0.4)),
        2,
        2,
        2,
        2,
    )


def planner_transition() -> List[Float32]:
    # Existing planner parity fixture in (new, old, theta, action) order.
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


def planner_transition_single_static() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def planner_observation() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def planner_theta_goal() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.15, 0.75))
    result.extend(pair(0.85, 0.25))
    return result^


def test_entropy_matches_jax_for_normalized_and_unnormalized_beliefs() raises:
    assert_close(entropy(pair(safe_log(0.25), safe_log(0.75))), 0.56233513)
    assert_close(
        entropy_unnormalized(pair(safe_log(2.0), safe_log(6.0))), 0.56233513
    )


def test_energy_skips_structural_zeros() raises:
    var region = pair(safe_log(0.25), safe_log(0.75))
    assert_close(energy(region, pair(safe_log(0.8), safe_log(0.2))), 1.2628644)
    assert_close(energy(region, pair(safe_log(0.8), LOG_ZERO)), 0.05578589)


def test_conditional_entropy_helpers_match_jax() raises:
    var dynamic_regions = dyn_regions()
    var observation_regions = log_values(obs_region_probabilities())
    assert_close(action_cond_entropy(dynamic_regions, 2, 2, 2, 2), 1.3838545084)
    assert_close(dyn_cond_entropy(dynamic_regions, 2, 2, 2, 2), 1.3459075689)
    assert_close(
        obs_cond_entropy(observation_regions, 2, 1, 2, 2, 2), 1.9480485916
    )
    assert_true(action_cond_entropy(dynamic_regions, 2, 2, 2, 2) >= 0.0)
    assert_true(dyn_cond_entropy(dynamic_regions, 2, 2, 2, 2) >= 0.0)


def test_bethe_vfe_matches_jax() raises:
    assert_close(bethe_fixture(), -3.5446314812)


def test_region_channel_precise_and_active_vfes_match_jax() raises:
    var log_transition = log_values(transition())
    var reduced = reduced_placeholder()
    var fwd = fwd_messages()
    var bwd = bwd_messages()
    var q_u = actions()
    var cavities = dyn_cavities()
    var prior = log_values(pair(0.5, 0.5))
    var action_prior = log_values(pair(0.6, 0.4))
    var dynamic_regions = dyn_regions()
    var observation_regions = log_values(obs_region_probabilities())
    assert_close(
        compute_region_extended_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            cavities,
            prior,
            action_prior,
            dynamic_regions,
            observation_regions,
            2,
            2,
            2,
            2,
            1,
            2,
        ),
        -2.9424905777,
    )
    assert_close(
        compute_dyn_channel_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            cavities,
            prior,
            action_prior,
            dynamic_regions,
            2,
            2,
            2,
            2,
        ),
        -4.8905391693,
    )
    assert_close(
        compute_vbp_channel_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            cavities,
            prior,
            action_prior,
            dynamic_regions,
            2,
            2,
            2,
            2,
        ),
        -2.1607770920,
    )
    assert_close(
        compute_precise_info_seeking_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            cavities,
            prior,
            action_prior,
            dynamic_regions,
            observation_regions,
            2,
            2,
            2,
            2,
            1,
            2,
        ),
        -0.2127285004,
    )
    assert_close(
        compute_active_inference_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            cavities,
            prior,
            action_prior,
            dynamic_regions,
            observation_regions,
            2,
            2,
            2,
            2,
            1,
            2,
        ),
        -1.5586360693,
    )


def test_nuijten_vfe_matches_jax_probability_region_contract() raises:
    assert_close(
        compute_nuijten_vfe(
            dyn_regions(),
            obs_region_probabilities(),
            fwd_messages(),
            bwd_messages(),
            actions(),
            tiled_transition_kernel(),
            observation_factor(),
            dyn_cavities(),
            obs_cavities(),
            log_values(pair(0.5, 0.5)),
            action_prior_per_t(),
            2,
            2,
            2,
            2,
            1,
            2,
        ),
        2.5995984077,
    )


def planner_loopy(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    return loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        goal,
        pair(0.6, 0.4),
        theta_goal,
        2,
        2,
        2,
        2,
        2,
    )


def planner_region(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    return region_extended_loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        goal,
        pair(0.6, 0.4),
        theta_goal,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def planner_nuijten(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    return nuijten_mp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        goal,
        pair(0.6, 0.4),
        theta_goal,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )


def planner_dyn(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    return dyn_channel_loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        goal,
        pair(0.6, 0.4),
        theta_goal,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def planner_vbp(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    return vbp_channel_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        goal,
        pair(0.6, 0.4),
        theta_goal,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def planner_precise(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    return precise_info_seeking_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        goal,
        pair(0.6, 0.4),
        theta_goal,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def planner_active(goal: List[Float32], theta_goal: Bool) -> List[Float32]:
    return active_inference_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        goal,
        pair(0.6, 0.4),
        theta_goal,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )


def test_loopy_terminal_and_theta_traces_match_jax() raises:
    var terminal = planner_loopy(pair(0.15, 0.85), False)
    assert_true(len(terminal) == 4)
    assert_close(terminal[0], 0.5944433212)
    assert_close(terminal[1], 0.4055566490)
    assert_close(terminal[2], -3.8262903690)
    assert_close(terminal[3], -3.8269684315)
    var theta = planner_loopy(planner_theta_goal(), True)
    assert_close(theta[0], 0.5906499028)
    assert_close(theta[1], 0.4093501568)
    assert_close(theta[2], -4.0740485191)
    assert_close(theta[3], -4.0746631622)


def test_explicit_goal_mode_disambiguates_single_static_state() raises:
    var static = List[Float32]()
    static.append(1.0)
    var terminal = loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        static,
        planner_transition_single_static(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        False,
        2,
        2,
        2,
        2,
        1,
    )
    var theta = loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        static,
        planner_transition_single_static(),
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        True,
        2,
        2,
        2,
        2,
        1,
    )
    assert_close(terminal[0], 0.5874962211)
    assert_close(terminal[2], -3.1058092117)
    assert_close(theta[0], 0.6761301756)
    assert_close(theta[2], -3.2784326077)


def test_region_terminal_and_theta_traces_match_jax() raises:
    var terminal = planner_region(pair(0.15, 0.85), False)
    assert_true(len(terminal) == 44)
    assert_close(terminal[0], 0.5999398828)
    assert_close(terminal[1], 0.4000601470)
    assert_close(terminal[2], -0.4801756144)
    assert_close(terminal[17], -0.3857865632)
    assert_close(terminal[18], -0.0006803344)
    assert_close(terminal[41], -4.7250075340)
    assert_close(terminal[42], -6.4673976898)
    assert_close(terminal[43], -7.0882754326)
    var theta = planner_region(planner_theta_goal(), True)
    assert_close(theta[0], 0.5947961211)
    assert_close(theta[1], 0.4052039385)
    assert_close(theta[2], -0.5363883972)
    assert_close(theta[17], -0.6696937084)
    assert_close(theta[42], -7.0286130905)
    assert_close(theta[43], -7.4308137894)


def test_nuijten_terminal_and_theta_traces_match_jax() raises:
    var terminal = planner_nuijten(pair(0.15, 0.85), False)
    assert_true(len(terminal) == 60)
    assert_close(terminal[0], 0.5128134489)
    assert_close(terminal[1], 0.4871865511)
    assert_close(terminal[2], -3.8468985558)
    assert_close(terminal[33], -4.1090788841)
    assert_close(terminal[34], 0.2935514450)
    assert_close(terminal[57], 0.0782248452)
    assert_close(terminal[58], 0.4731526375)
    assert_close(terminal[59], 0.4360868931)
    var theta = planner_nuijten(planner_theta_goal(), True)
    assert_close(theta[0], 0.5200071931)
    assert_close(theta[1], 0.4799928367)
    assert_close(theta[2], -5.5011434555)
    assert_close(theta[33], -5.0147757530)
    assert_close(theta[58], -0.1365780830)
    assert_close(theta[59], -0.1460067034)


def test_dyn_channel_terminal_and_theta_traces_match_jax() raises:
    var terminal = planner_dyn(pair(0.15, 0.85), False)
    assert_true(len(terminal) == 20)
    assert_close(terminal[0], 0.5961717367)
    assert_close(terminal[1], 0.4038282931)
    assert_close(terminal[2], -0.5891003609)
    assert_close(terminal[17], -0.3216637373)
    assert_close(terminal[18], -4.9266700745)
    assert_close(terminal[19], -5.1136736870)
    var theta = planner_dyn(planner_theta_goal(), True)
    assert_close(theta[0], 0.5982998013)
    assert_close(theta[1], 0.4017002583)
    assert_close(theta[2], -0.5594892502)
    assert_close(theta[17], -0.6102197766)
    assert_close(theta[18], -5.4284868240)
    assert_close(theta[19], -5.4047336578)


def test_vbp_channel_terminal_and_theta_traces_match_jax() raises:
    var terminal = planner_vbp(pair(0.15, 0.85), False)
    assert_true(len(terminal) == 12)
    assert_close(terminal[0], 0.6385504007)
    assert_close(terminal[1], 0.3614496291)
    assert_close(terminal[2], -0.5282676816)
    assert_close(terminal[9], -0.9548379779)
    assert_close(terminal[10], -2.4716849327)
    assert_close(terminal[11], -2.4838523865)
    var theta = planner_vbp(planner_theta_goal(), True)
    assert_close(theta[0], 0.6404780149)
    assert_close(theta[1], 0.3595220447)
    assert_close(theta[2], -0.5313208699)
    assert_close(theta[9], -0.9201439619)
    assert_close(theta[10], -2.7220985889)
    assert_close(theta[11], -2.7190852165)


def test_precise_terminal_and_theta_traces_match_jax() raises:
    var terminal = planner_precise(pair(0.15, 0.85), False)
    assert_true(len(terminal) == 36)
    assert_close(terminal[0], 0.6587315798)
    assert_close(terminal[1], 0.3412684798)
    assert_close(terminal[2], -0.4891316593)
    assert_close(terminal[9], -0.9514805079)
    assert_close(terminal[10], -0.0006803344)
    assert_close(terminal[33], -4.7250084877)
    assert_close(terminal[34], -3.9500231743)
    assert_close(terminal[35], -4.4422550201)
    var theta = planner_precise(planner_theta_goal(), True)
    assert_close(theta[0], 0.6469761729)
    assert_close(theta[1], 0.3530238271)
    assert_close(theta[2], -0.5107262135)
    assert_close(theta[9], -0.9162744284)
    assert_close(theta[34], -4.3019685745)
    assert_close(theta[35], -4.7070484161)


def test_active_terminal_and_theta_traces_match_jax() raises:
    var terminal = planner_active(pair(0.15, 0.85), False)
    assert_true(len(terminal) == 44)
    assert_close(terminal[0], 0.6473624706)
    assert_close(terminal[1], 0.3526374996)
    assert_close(terminal[2], -0.4812748432)
    assert_close(terminal[17], -0.3857866526)
    assert_close(terminal[18], -0.0006803344)
    assert_close(terminal[41], -4.7250080109)
    assert_close(terminal[42], -5.1213741302)
    assert_close(terminal[43], -5.7412571907)
    var theta = planner_active(planner_theta_goal(), True)
    assert_close(theta[0], 0.6425644755)
    assert_close(theta[1], 0.3574355543)
    assert_close(theta[2], -0.5365261436)
    assert_close(theta[17], -0.6696937084)
    assert_close(theta[42], -5.6825900078)
    assert_close(theta[43], -6.0809140205)


def test_trace_wrappers_preserve_action_masks() raises:
    var result = loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        False,
        1,
        1,
        2,
        2,
        2,
    )
    assert_close(result[0], 1.0)
    assert_close(result[1], 0.0)
    result = region_extended_loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        False,
        1,
        1,
        0.25,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_close(result[1], 0.0)
    result = nuijten_mp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        False,
        1,
        1,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_close(result[1], 0.0)
    result = dyn_channel_loopy_bp_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        False,
        1,
        1,
        0.25,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_close(result[1], 0.0)
    result = vbp_channel_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        False,
        1,
        1,
        0.25,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_close(result[1], 0.0)
    result = precise_info_seeking_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        False,
        1,
        1,
        0.25,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_close(result[1], 0.0)
    result = active_inference_planning_with_vfe(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        planner_transition(),
        planner_observation(),
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        False,
        1,
        1,
        0.25,
        2,
        2,
        2,
        1,
        2,
    )
    assert_close(result[0], 1.0)
    assert_close(result[1], 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
