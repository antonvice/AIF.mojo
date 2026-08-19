from std.collections import List

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
    dyn_cond_entropy,
    dyn_channel_loopy_bp_planning_with_vfe,
    loopy_bp_planning_with_vfe,
    nuijten_mp_planning_with_vfe,
    obs_cond_entropy,
    precise_info_seeking_planning_with_vfe,
    region_extended_loopy_bp_planning_with_vfe,
    vbp_channel_planning_with_vfe,
)
from aif_mojo.numerics import safe_log


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


def zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
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
    var probabilities = List[Float32]()
    for value in range(1, 17):
        probabilities.append(Float32(value) / 136.0)
    for value in range(17, 33):
        probabilities.append(Float32(value) / 392.0)
    return log_values(probabilities^)


def obs_region_probabilities() -> List[Float32]:
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
    var source = log_values(transition())
    var result = List[Float32]()
    for _ in range(2):
        for old_idx in range(2):
            for new_idx in range(2):
                for static_idx in range(2):
                    for action_idx in range(2):
                        var offset = (
                            (new_idx * 2 + old_idx) * 2 + static_idx
                        ) * 2 + action_idx
                        result.append(source[offset])
    return result^


def planner_transition() -> List[Float32]:
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


def emit_trace(name: String, values: List[Float32]):
    for index in range(len(values)):
        print("convergence_trace", name, index, values[index])


def emit_planner_traces():
    var state = pair(0.65, 0.35)
    var static = pair(0.55, 0.45)
    var terminal = pair(0.15, 0.85)
    var theta = planner_theta_goal()
    var actions = pair(0.6, 0.4)
    emit_trace(
        "loopy_terminal",
        loopy_bp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            terminal,
            actions,
            False,
            2,
            2,
            2,
            2,
            2,
        ),
    )
    emit_trace(
        "loopy_theta",
        loopy_bp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            theta,
            actions,
            True,
            2,
            2,
            2,
            2,
            2,
        ),
    )
    emit_trace(
        "region_terminal",
        region_extended_loopy_bp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            terminal,
            actions,
            False,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "region_theta",
        region_extended_loopy_bp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            theta,
            actions,
            True,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "nuijten_terminal",
        nuijten_mp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            terminal,
            actions,
            False,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "nuijten_theta",
        nuijten_mp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            theta,
            actions,
            True,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "dyn_terminal",
        dyn_channel_loopy_bp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            terminal,
            actions,
            False,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "dyn_theta",
        dyn_channel_loopy_bp_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            theta,
            actions,
            True,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "vbp_terminal",
        vbp_channel_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            terminal,
            actions,
            False,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "vbp_theta",
        vbp_channel_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            theta,
            actions,
            True,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "precise_terminal",
        precise_info_seeking_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            terminal,
            actions,
            False,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "precise_theta",
        precise_info_seeking_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            theta,
            actions,
            True,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "active_terminal",
        active_inference_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            terminal,
            actions,
            False,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_trace(
        "active_theta",
        active_inference_planning_with_vfe(
            state,
            static,
            planner_transition(),
            planner_observation(),
            theta,
            actions,
            True,
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        ),
    )


def main():
    var log_transition = log_values(transition())
    var reduced = zeros(16)
    var fwd = fwd_messages()
    var bwd = bwd_messages()
    var q_u = actions()
    var dyn_cavity = dyn_cavities()
    var theta_prior = log_values(pair(0.5, 0.5))
    var action_prior = log_values(pair(0.6, 0.4))
    var dynamic_regions = dyn_regions()
    var observation_probabilities = obs_region_probabilities()
    var observation_regions = log_values(observation_probabilities)
    print(
        "convergence_vfe",
        action_cond_entropy(dynamic_regions, 2, 2, 2, 2),
    )
    print(
        "convergence_vfe",
        dyn_cond_entropy(dynamic_regions, 2, 2, 2, 2),
    )
    print(
        "convergence_vfe",
        obs_cond_entropy(observation_regions, 2, 1, 2, 2, 2),
    )
    print(
        "convergence_vfe",
        compute_bethe_vfe_loopy(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            dyn_cavity,
            theta_prior,
            action_prior,
            2,
            2,
            2,
            2,
        ),
    )
    print(
        "convergence_vfe",
        compute_region_extended_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            dyn_cavity,
            theta_prior,
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
    )
    print(
        "convergence_vfe",
        compute_dyn_channel_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            dyn_cavity,
            theta_prior,
            action_prior,
            dynamic_regions,
            2,
            2,
            2,
            2,
        ),
    )
    print(
        "convergence_vfe",
        compute_vbp_channel_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            dyn_cavity,
            theta_prior,
            action_prior,
            dynamic_regions,
            2,
            2,
            2,
            2,
        ),
    )
    print(
        "convergence_vfe",
        compute_precise_info_seeking_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            dyn_cavity,
            theta_prior,
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
    )
    print(
        "convergence_vfe",
        compute_active_inference_vfe(
            log_transition,
            reduced,
            fwd,
            bwd,
            q_u,
            dyn_cavity,
            theta_prior,
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
    )
    var action_prior_time = List[Float32]()
    action_prior_time.extend(pair(0.55, 0.45))
    action_prior_time.extend(pair(0.2, 0.8))
    print(
        "convergence_vfe",
        compute_nuijten_vfe(
            dynamic_regions,
            observation_probabilities,
            fwd,
            bwd,
            q_u,
            tiled_transition_kernel(),
            observation_factor(),
            dyn_cavity,
            obs_cavities(),
            theta_prior,
            action_prior_time,
            2,
            2,
            2,
            2,
            1,
            2,
        ),
    )
    emit_planner_traces()
