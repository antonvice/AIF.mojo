from std.collections import List

from aif_mojo.active_inference import (
    _compute_dense_log_base,
    active_inference_planning_dense,
    active_inference_planning_dense_theta_goal,
    active_inference_planning_precomputed,
    active_inference_planning_precomputed_theta_goal,
    compute_dyn_kernels_aif,
    precompute_obs_channels,
    precompute_pref_to_x,
)
from aif_mojo.messages import (
    backward_message_2d,
    backward_message_3d,
    backward_message_to_other_3d,
    combine_messages,
    combine_messages_log,
    forward_message_2d,
    forward_message_3d,
    forward_message_4d,
    marginalize_static,
)
from aif_mojo.loopy_bp import (
    backward_messages,
    compute_action_marginals,
    compute_reduced_per_t_dense,
    compute_theta_cavities,
    forward_pass,
    loopy_bp_planning_sparse,
    loopy_bp_planning_sparse_theta_goal,
    loopy_bp_planning_dense,
    loopy_bp_planning_dense_theta_goal,
)
from aif_mojo.loopy_vbp import (
    backward_pass_vbp,
    compute_dyn_to_theta_vbp,
    forward_pass_vbp,
    loopy_vbp_planning_dense,
    loopy_vbp_planning_dense_theta_goal,
)
from aif_mojo.convergence import energy, entropy, entropy_unnormalized
from aif_mojo.dyn_channel_loopy_bp import (
    dyn_channel_loopy_bp_planning_dense,
    dyn_channel_loopy_bp_planning_dense_theta_goal,
    dyn_channel_loopy_bp_planning_sparse,
    dyn_channel_loopy_bp_planning_sparse_theta_goal,
)
from aif_mojo.numerics import LOG_ZERO, safe_log, safe_log_div, softmax
from aif_mojo.nuijten_mp import (
    nuijten_mp_planning_dense,
    nuijten_mp_planning_dense_theta_goal,
    nuijten_mp_planning_sparse,
    nuijten_mp_planning_sparse_theta_goal,
)
from aif_mojo.precise_info_seeking import (
    compute_precise_obs_kernels,
    precise_info_seeking_planning_dense,
    precise_info_seeking_planning_dense_theta_goal,
)
from aif_mojo.region_extended_loopy_bp import (
    region_extended_loopy_bp_planning_dense,
    region_extended_loopy_bp_planning_dense_theta_goal,
)
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
from aif_mojo.state_inference import (
    state_inference_step,
    state_inference_step_sparse,
)
from aif_mojo.vbp_channel import (
    vbp_channel_planning_dense,
    vbp_channel_planning_dense_theta_goal,
    vbp_channel_planning_sparse,
    vbp_channel_planning_sparse_theta_goal,
)


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def transition_indices() -> List[Int]:
    var result = List[Int]()
    result.append(0)
    result.append(1)
    result.append(1)
    result.append(1)
    return result^


def two_state_transition_indices() -> List[Int]:
    var result = List[Int]()
    result.append(0)
    result.append(1)
    result.append(1)
    result.append(0)
    result.append(1)
    result.append(0)
    result.append(0)
    result.append(1)
    return result^


def flip_or_stay_reduced(horizon: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(horizon):
        result.append(0.0)
        result.append(safe_log(0.0))
        result.append(safe_log(0.0))
        result.append(0.0)
        result.append(safe_log(0.0))
        result.append(0.0)
        result.append(0.0)
        result.append(safe_log(0.0))
    return result^


def dense_uncertain_transition() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(1.0, 0.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(1.0, 0.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(1.0, 0.0))
    result.extend(pair(1.0, 0.0))
    result.extend(pair(0.0, 1.0))
    return result^


def loopy_vbp_transition() -> List[Float32]:
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


def log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
    return result^


def main():
    var impossible_softmax = softmax(pair(LOG_ZERO, LOG_ZERO))
    print("softmax_impossible", impossible_softmax[0], impossible_softmax[1])

    var tensor2 = List[Float32]()
    tensor2.extend(pair(0.8, 0.1))
    tensor2.extend(pair(0.2, 0.9))
    var forward2 = forward_message_2d(tensor2, 2, 2, pair(0.25, 0.75))
    print("forward2d", forward2[0], forward2[1])
    var backward2 = backward_message_2d(tensor2, 2, 2, pair(1.0, 0.0))
    print("backward2d", backward2[0], backward2[1])

    var xor3 = List[Float32]()
    xor3.extend(pair(1.0, 0.0))
    xor3.extend(pair(0.0, 1.0))
    xor3.extend(pair(0.0, 1.0))
    xor3.extend(pair(1.0, 0.0))
    var forward3 = forward_message_3d(
        xor3, 2, 2, 2, pair(0.25, 0.75), pair(0.6, 0.4)
    )
    print("forward3d", forward3[0], forward3[1])
    var backward3 = backward_message_3d(
        xor3, 2, 2, 2, pair(1.0, 0.0), pair(0.6, 0.4)
    )
    print("backward3d", backward3[0], backward3[1])
    var backward_other = backward_message_to_other_3d(
        xor3, 2, 2, 2, pair(1.0, 0.0), pair(0.25, 0.75)
    )
    print("backward_other3d", backward_other[0], backward_other[1])

    var parity4 = List[Float32]()
    parity4.extend(pair(1.0, 0.0))
    parity4.extend(pair(0.0, 1.0))
    parity4.extend(pair(0.0, 1.0))
    parity4.extend(pair(1.0, 0.0))
    parity4.extend(pair(0.0, 1.0))
    parity4.extend(pair(1.0, 0.0))
    parity4.extend(pair(1.0, 0.0))
    parity4.extend(pair(0.0, 1.0))
    var forward4 = forward_message_4d(
        parity4,
        2,
        2,
        2,
        2,
        pair(0.2, 0.8),
        pair(0.3, 0.7),
        pair(0.4, 0.6),
    )
    print("forward4d", forward4[0], forward4[1])

    var messages = List[Float32]()
    messages.extend(pair(0.8, 0.2))
    messages.extend(pair(0.6, 0.4))
    var combined = combine_messages(messages, 2, 2)
    print("combined", combined[0], combined[1])
    var log_messages = log_values(messages)
    var combined_log = combine_messages_log(log_messages, 2, 2)
    print("combined_log", combined_log[0], combined_log[1])
    print(
        "safe_log_edges",
        safe_log(0.0),
        safe_log(1.0e-30),
        safe_log(0.5),
        safe_log(-1.0),
        safe_log_div(safe_log(0.5), safe_log(0.25)),
        safe_log_div(LOG_ZERO, safe_log(0.25)),
        safe_log_div(safe_log(0.5), LOG_ZERO),
        safe_log_div(LOG_ZERO, LOG_ZERO),
    )

    var log_tensor = List[Float32]()
    log_tensor.extend(pair(safe_log(0.9), safe_log(0.2)))
    log_tensor.extend(pair(safe_log(0.1), safe_log(0.8)))
    var marginalized = marginalize_static(
        log_tensor,
        2,
        1,
        2,
        1,
        pair(safe_log(0.25), safe_log(0.75)),
    )
    print("marginalized", marginalized[0], marginalized[1])

    var sparse_base = compute_log_base_sparse(
        transition_indices(), pair(safe_log(0.25), safe_log(0.75)), 2, 1, 2
    )
    print(
        "sparse_base",
        sparse_base[0],
        sparse_base[1],
        sparse_base[2],
        sparse_base[3],
    )

    var cavities = List[Float32]()
    cavities.extend(pair(safe_log(0.25), safe_log(0.75)))
    cavities.extend(pair(safe_log(0.6), safe_log(0.4)))
    var reduced = sparse_reduced(transition_indices(), cavities, 2, 2, 1, 2)
    print(
        "sparse_reduced",
        reduced[0],
        reduced[1],
        reduced[2],
        reduced[3],
        reduced[4],
        reduced[5],
        reduced[6],
        reduced[7],
    )

    var weighted = sparse_reduced_weighted(
        transition_indices(),
        pair(safe_log(0.25), safe_log(0.75)),
        pair(safe_log(0.5), safe_log(0.2)),
        1,
        2,
        1,
        2,
    )
    print("sparse_weighted", weighted[0], weighted[1], weighted[2], weighted[3])
    var dyn_channels = List[Float32]()
    dyn_channels.extend(pair(safe_log(0.4), safe_log(0.6)))
    dyn_channels.extend(pair(safe_log(0.2), safe_log(0.8)))
    var action_channels = List[Float32]()
    action_channels.append(0.0)
    action_channels.append(0.0)
    var dyn_channel_reduced = sparse_reduced_dyn_channel(
        transition_indices(),
        pair(safe_log(0.25), safe_log(0.75)),
        dyn_channels,
        action_channels,
        1,
        2,
        1,
        2,
    )
    print(
        "sparse_dyn_channel_reduced",
        dyn_channel_reduced[0],
        dyn_channel_reduced[1],
        dyn_channel_reduced[2],
        dyn_channel_reduced[3],
    )

    var sparse_fwd = List[Float32]()
    sparse_fwd.extend(pair(safe_log(0.6), safe_log(0.4)))
    sparse_fwd.extend(pair(safe_log(0.5), safe_log(0.5)))
    var sparse_bwd = List[Float32]()
    sparse_bwd.extend(pair(safe_log(0.5), safe_log(0.5)))
    sparse_bwd.extend(pair(safe_log(0.7), safe_log(0.3)))
    var sparse_local = List[Float32]()
    for _ in range(4):
        sparse_local.append(0.0)
    var single_action = List[Float32]()
    single_action.append(0.0)
    var sparse_theta = sparse_dyn_to_theta(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        single_action,
        1,
        2,
        1,
        2,
    )
    print("sparse_theta", sparse_theta[0], sparse_theta[1])
    var sparse_theta_weighted = sparse_dyn_to_theta_weighted(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        single_action,
        pair(safe_log(0.5), safe_log(0.2)),
        1,
        2,
        1,
        2,
    )
    print(
        "sparse_theta_weighted",
        sparse_theta_weighted[0],
        sparse_theta_weighted[1],
    )
    var sparse_pair = sparse_pair_marginal(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        pair(safe_log(0.25), safe_log(0.75)),
        single_action,
        1,
        2,
        1,
        2,
    )
    print("sparse_pair", sparse_pair[0], sparse_pair[1])
    var sparse_pair_weighted = sparse_pair_marginal_weighted(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        pair(safe_log(0.25), safe_log(0.75)),
        single_action,
        pair(safe_log(0.5), safe_log(0.2)),
        1,
        2,
        1,
        2,
    )
    print(
        "sparse_pair_weighted",
        sparse_pair_weighted[0],
        sparse_pair_weighted[1],
    )
    var dyn_theta = sparse_dyn_to_theta_dyn_channel(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        single_action,
        dyn_channels,
        action_channels,
        1,
        2,
        1,
        2,
    )
    print("sparse_dyn_channel_theta", dyn_theta[0], dyn_theta[1])
    var dyn_channel_pair = sparse_dyn_channels_and_pair_dyn_channel(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        pair(safe_log(0.25), safe_log(0.75)),
        single_action,
        dyn_channels,
        action_channels,
        1,
        2,
        1,
        2,
    )
    print(
        "sparse_dyn_channel_pair",
        dyn_channel_pair[0],
        dyn_channel_pair[1],
        dyn_channel_pair[2],
        dyn_channel_pair[3],
        dyn_channel_pair[4],
        dyn_channel_pair[5],
    )
    var channel_pair = sparse_dyn_channels_and_pair(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        pair(safe_log(0.25), safe_log(0.75)),
        single_action,
        1,
        2,
        1,
        2,
    )
    print(
        "sparse_channel_pair",
        channel_pair[0],
        channel_pair[1],
        channel_pair[2],
        channel_pair[3],
        channel_pair[4],
        channel_pair[5],
    )
    var weighted_channel_pair = sparse_dyn_channels_and_pair_weighted(
        transition_indices(),
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        pair(safe_log(0.25), safe_log(0.75)),
        single_action,
        pair(safe_log(0.5), safe_log(0.2)),
        1,
        2,
        1,
        2,
    )
    print(
        "sparse_weighted_channel_pair",
        weighted_channel_pair[0],
        weighted_channel_pair[1],
        weighted_channel_pair[2],
        weighted_channel_pair[3],
        weighted_channel_pair[4],
        weighted_channel_pair[5],
    )
    var efe_transitions = List[Int]()
    efe_transitions.append(0)
    efe_transitions.append(1)
    efe_transitions.append(1)
    efe_transitions.append(0)
    efe_transitions.append(1)
    efe_transitions.append(0)
    efe_transitions.append(0)
    efe_transitions.append(1)
    var efe_unmasked = sparse_efe_action_prior(
        efe_transitions,
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        pair(safe_log(0.7), safe_log(0.3)),
        pair(safe_log(0.5), safe_log(0.5)),
        pair(1.0, 1.0),
        1,
        2,
        2,
        2,
    )
    print("sparse_efe", efe_unmasked[0], efe_unmasked[1])
    var efe_masked = sparse_efe_action_prior(
        efe_transitions,
        sparse_fwd,
        sparse_bwd,
        sparse_local,
        pair(safe_log(0.7), safe_log(0.3)),
        pair(safe_log(0.5), safe_log(0.5)),
        pair(1.0, 0.0),
        1,
        2,
        2,
        2,
    )
    print("sparse_efe_masked", efe_masked[0], efe_masked[1])

    var dyn_plan_observation = List[Float32]()
    dyn_plan_observation.append(0.9)
    dyn_plan_observation.append(0.2)
    dyn_plan_observation.append(0.3)
    dyn_plan_observation.append(0.8)
    dyn_plan_observation.append(0.1)
    dyn_plan_observation.append(0.8)
    dyn_plan_observation.append(0.7)
    dyn_plan_observation.append(0.2)
    var dyn_channel_plan = dyn_channel_loopy_bp_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        efe_transitions,
        dyn_plan_observation,
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "dyn_channel_plan",
        dyn_channel_plan[0],
        dyn_channel_plan[1],
        dyn_channel_plan[2],
        dyn_channel_plan[3],
        dyn_channel_plan[4],
        dyn_channel_plan[5],
        dyn_channel_plan[6],
        dyn_channel_plan[7],
        dyn_channel_plan[8],
        dyn_channel_plan[9],
        dyn_channel_plan[10],
        dyn_channel_plan[11],
        dyn_channel_plan[12],
        dyn_channel_plan[13],
        dyn_channel_plan[14],
        dyn_channel_plan[15],
        dyn_channel_plan[16],
        dyn_channel_plan[17],
    )
    var dyn_channel_theta_goal = List[Float32]()
    dyn_channel_theta_goal.append(0.1)
    dyn_channel_theta_goal.append(0.8)
    dyn_channel_theta_goal.append(0.9)
    dyn_channel_theta_goal.append(0.2)
    var dyn_channel_theta_plan = (
        dyn_channel_loopy_bp_planning_sparse_theta_goal(
            pair(0.8, 0.2),
            pair(0.6, 0.4),
            efe_transitions,
            dyn_plan_observation,
            dyn_channel_theta_goal,
            pair(0.55, 0.45),
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        )
    )
    print(
        "dyn_channel_theta_plan",
        dyn_channel_theta_plan[0],
        dyn_channel_theta_plan[1],
        dyn_channel_theta_plan[2],
        dyn_channel_theta_plan[3],
        dyn_channel_theta_plan[4],
        dyn_channel_theta_plan[5],
        dyn_channel_theta_plan[6],
        dyn_channel_theta_plan[7],
        dyn_channel_theta_plan[8],
        dyn_channel_theta_plan[9],
        dyn_channel_theta_plan[10],
        dyn_channel_theta_plan[11],
        dyn_channel_theta_plan[12],
        dyn_channel_theta_plan[13],
        dyn_channel_theta_plan[14],
        dyn_channel_theta_plan[15],
        dyn_channel_theta_plan[16],
        dyn_channel_theta_plan[17],
    )
    var dyn_channel_masked_plan = dyn_channel_loopy_bp_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        efe_transitions,
        dyn_plan_observation,
        pair(0.1, 0.9),
        pair(1.0, 0.0),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "dyn_channel_masked_plan",
        dyn_channel_masked_plan[0],
        dyn_channel_masked_plan[1],
    )
    var dyn_channel_dense_plan = dyn_channel_loopy_bp_planning_dense(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_uncertain_transition(),
        dyn_plan_observation,
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "dyn_channel_dense_plan",
        dyn_channel_dense_plan[0],
        dyn_channel_dense_plan[1],
        dyn_channel_dense_plan[2],
        dyn_channel_dense_plan[3],
        dyn_channel_dense_plan[4],
        dyn_channel_dense_plan[5],
        dyn_channel_dense_plan[6],
        dyn_channel_dense_plan[7],
        dyn_channel_dense_plan[8],
        dyn_channel_dense_plan[9],
        dyn_channel_dense_plan[10],
        dyn_channel_dense_plan[11],
        dyn_channel_dense_plan[12],
        dyn_channel_dense_plan[13],
        dyn_channel_dense_plan[14],
        dyn_channel_dense_plan[15],
        dyn_channel_dense_plan[16],
        dyn_channel_dense_plan[17],
    )
    var dyn_channel_dense_theta_plan = (
        dyn_channel_loopy_bp_planning_dense_theta_goal(
            pair(0.8, 0.2),
            pair(0.6, 0.4),
            dense_uncertain_transition(),
            dyn_plan_observation,
            dyn_channel_theta_goal,
            pair(0.55, 0.45),
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
        )
    )
    print(
        "dyn_channel_dense_theta_plan",
        dyn_channel_dense_theta_plan[0],
        dyn_channel_dense_theta_plan[1],
        dyn_channel_dense_theta_plan[2],
        dyn_channel_dense_theta_plan[3],
        dyn_channel_dense_theta_plan[4],
        dyn_channel_dense_theta_plan[5],
        dyn_channel_dense_theta_plan[6],
        dyn_channel_dense_theta_plan[7],
        dyn_channel_dense_theta_plan[8],
        dyn_channel_dense_theta_plan[9],
        dyn_channel_dense_theta_plan[10],
        dyn_channel_dense_theta_plan[11],
        dyn_channel_dense_theta_plan[12],
        dyn_channel_dense_theta_plan[13],
        dyn_channel_dense_theta_plan[14],
        dyn_channel_dense_theta_plan[15],
        dyn_channel_dense_theta_plan[16],
        dyn_channel_dense_theta_plan[17],
    )
    var vbp_sparse_plan = vbp_channel_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        efe_transitions,
        dyn_plan_observation,
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "vbp_sparse_plan",
        vbp_sparse_plan[0],
        vbp_sparse_plan[1],
        vbp_sparse_plan[2],
        vbp_sparse_plan[3],
        vbp_sparse_plan[4],
        vbp_sparse_plan[5],
        vbp_sparse_plan[6],
        vbp_sparse_plan[7],
        vbp_sparse_plan[8],
        vbp_sparse_plan[9],
    )
    var vbp_dense_plan = vbp_channel_planning_dense(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_uncertain_transition(),
        dyn_plan_observation,
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "vbp_dense_plan",
        vbp_dense_plan[0],
        vbp_dense_plan[1],
        vbp_dense_plan[2],
        vbp_dense_plan[3],
        vbp_dense_plan[4],
        vbp_dense_plan[5],
        vbp_dense_plan[6],
        vbp_dense_plan[7],
        vbp_dense_plan[8],
        vbp_dense_plan[9],
    )
    var vbp_sparse_theta_plan = vbp_channel_planning_sparse_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        efe_transitions,
        dyn_plan_observation,
        dyn_channel_theta_goal,
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "vbp_sparse_theta_plan",
        vbp_sparse_theta_plan[0],
        vbp_sparse_theta_plan[1],
        vbp_sparse_theta_plan[2],
        vbp_sparse_theta_plan[3],
        vbp_sparse_theta_plan[4],
        vbp_sparse_theta_plan[5],
        vbp_sparse_theta_plan[6],
        vbp_sparse_theta_plan[7],
        vbp_sparse_theta_plan[8],
        vbp_sparse_theta_plan[9],
    )
    var vbp_dense_theta_plan = vbp_channel_planning_dense_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_uncertain_transition(),
        dyn_plan_observation,
        dyn_channel_theta_goal,
        pair(0.55, 0.45),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "vbp_dense_theta_plan",
        vbp_dense_theta_plan[0],
        vbp_dense_theta_plan[1],
        vbp_dense_theta_plan[2],
        vbp_dense_theta_plan[3],
        vbp_dense_theta_plan[4],
        vbp_dense_theta_plan[5],
        vbp_dense_theta_plan[6],
        vbp_dense_theta_plan[7],
        vbp_dense_theta_plan[8],
        vbp_dense_theta_plan[9],
    )
    var nuijten_sparse_plan = nuijten_mp_planning_sparse(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        efe_transitions,
        dyn_plan_observation,
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "nuijten_sparse_plan",
        nuijten_sparse_plan[0],
        nuijten_sparse_plan[1],
        nuijten_sparse_plan[2],
        nuijten_sparse_plan[3],
        nuijten_sparse_plan[4],
        nuijten_sparse_plan[5],
        nuijten_sparse_plan[6],
        nuijten_sparse_plan[7],
        nuijten_sparse_plan[8],
        nuijten_sparse_plan[9],
        nuijten_sparse_plan[10],
        nuijten_sparse_plan[11],
        nuijten_sparse_plan[12],
        nuijten_sparse_plan[13],
        nuijten_sparse_plan[14],
        nuijten_sparse_plan[15],
        nuijten_sparse_plan[16],
        nuijten_sparse_plan[17],
        nuijten_sparse_plan[18],
        nuijten_sparse_plan[19],
        nuijten_sparse_plan[20],
        nuijten_sparse_plan[21],
        nuijten_sparse_plan[22],
        nuijten_sparse_plan[23],
        nuijten_sparse_plan[24],
        nuijten_sparse_plan[25],
    )
    var nuijten_sparse_theta_plan = nuijten_mp_planning_sparse_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        efe_transitions,
        dyn_plan_observation,
        dyn_channel_theta_goal,
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "nuijten_sparse_theta_plan",
        nuijten_sparse_theta_plan[0],
        nuijten_sparse_theta_plan[1],
        nuijten_sparse_theta_plan[2],
        nuijten_sparse_theta_plan[3],
        nuijten_sparse_theta_plan[4],
        nuijten_sparse_theta_plan[5],
        nuijten_sparse_theta_plan[6],
        nuijten_sparse_theta_plan[7],
        nuijten_sparse_theta_plan[8],
        nuijten_sparse_theta_plan[9],
        nuijten_sparse_theta_plan[10],
        nuijten_sparse_theta_plan[11],
        nuijten_sparse_theta_plan[12],
        nuijten_sparse_theta_plan[13],
        nuijten_sparse_theta_plan[14],
        nuijten_sparse_theta_plan[15],
        nuijten_sparse_theta_plan[16],
        nuijten_sparse_theta_plan[17],
        nuijten_sparse_theta_plan[18],
        nuijten_sparse_theta_plan[19],
        nuijten_sparse_theta_plan[20],
        nuijten_sparse_theta_plan[21],
        nuijten_sparse_theta_plan[22],
        nuijten_sparse_theta_plan[23],
        nuijten_sparse_theta_plan[24],
        nuijten_sparse_theta_plan[25],
    )
    var nuijten_dense_plan = nuijten_mp_planning_dense(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_uncertain_transition(),
        dyn_plan_observation,
        pair(0.1, 0.9),
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "nuijten_dense_summary",
        nuijten_dense_plan[0],
        nuijten_dense_plan[1],
        nuijten_dense_plan[2],
        nuijten_dense_plan[33],
        nuijten_dense_plan[34],
        nuijten_dense_plan[57],
    )
    var nuijten_dense_theta_plan = nuijten_mp_planning_dense_theta_goal(
        pair(0.8, 0.2),
        pair(0.6, 0.4),
        dense_uncertain_transition(),
        dyn_plan_observation,
        dyn_channel_theta_goal,
        pair(0.55, 0.45),
        2,
        2,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "nuijten_dense_theta_summary",
        nuijten_dense_theta_plan[0],
        nuijten_dense_theta_plan[1],
        nuijten_dense_theta_plan[2],
        nuijten_dense_theta_plan[33],
        nuijten_dense_theta_plan[34],
        nuijten_dense_theta_plan[57],
    )

    var loopy_vbp_cavities = List[Float32]()
    loopy_vbp_cavities.extend(pair(safe_log(0.55), safe_log(0.45)))
    loopy_vbp_cavities.extend(pair(safe_log(0.55), safe_log(0.45)))
    var loopy_vbp_log_transition = log_values(loopy_vbp_transition())
    var loopy_vbp_reduced = compute_reduced_per_t_dense(
        loopy_vbp_log_transition, loopy_vbp_cavities, 2, 2, 2, 2
    )
    var loopy_vbp_backward = backward_pass_vbp(
        loopy_vbp_reduced,
        pair(safe_log(0.15), safe_log(0.85)),
        2,
        2,
        2,
    )
    print(
        "loopy_vbp_backward",
        loopy_vbp_backward[0],
        loopy_vbp_backward[1],
        loopy_vbp_backward[2],
        loopy_vbp_backward[3],
        loopy_vbp_backward[4],
        loopy_vbp_backward[5],
        loopy_vbp_backward[6],
        loopy_vbp_backward[7],
        loopy_vbp_backward[8],
        loopy_vbp_backward[9],
        loopy_vbp_backward[10],
        loopy_vbp_backward[11],
        loopy_vbp_backward[12],
        loopy_vbp_backward[13],
    )
    var loopy_vbp_q = List[Float32]()
    for index in range(8):
        loopy_vbp_q.append(loopy_vbp_backward[6 + index])
    var loopy_vbp_forward = forward_pass_vbp(
        loopy_vbp_reduced,
        pair(safe_log(0.65), safe_log(0.35)),
        loopy_vbp_q,
        2,
        2,
        2,
    )
    print(
        "loopy_vbp_forward",
        loopy_vbp_forward[0],
        loopy_vbp_forward[1],
        loopy_vbp_forward[2],
        loopy_vbp_forward[3],
        loopy_vbp_forward[4],
        loopy_vbp_forward[5],
    )
    var loopy_vbp_values = List[Float32]()
    for index in range(6):
        loopy_vbp_values.append(loopy_vbp_backward[index])
    var loopy_vbp_dyn = compute_dyn_to_theta_vbp(
        loopy_vbp_log_transition,
        loopy_vbp_forward,
        loopy_vbp_values,
        2,
        2,
        2,
        2,
    )
    print(
        "loopy_vbp_dyn_theta",
        loopy_vbp_dyn[0],
        loopy_vbp_dyn[1],
        loopy_vbp_dyn[2],
        loopy_vbp_dyn[3],
    )
    var loopy_vbp_plan = loopy_vbp_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        pair(0.15, 0.85),
        2,
        2,
        2,
        2,
        2,
    )
    print("loopy_vbp_plan", loopy_vbp_plan[0], loopy_vbp_plan[1])
    var loopy_vbp_theta_goal = List[Float32]()
    loopy_vbp_theta_goal.extend(pair(0.15, 0.75))
    loopy_vbp_theta_goal.extend(pair(0.85, 0.25))
    var loopy_vbp_theta_plan = loopy_vbp_planning_dense_theta_goal(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        loopy_vbp_theta_goal,
        2,
        2,
        2,
        2,
        2,
    )
    print(
        "loopy_vbp_theta_plan",
        loopy_vbp_theta_plan[0],
        loopy_vbp_theta_plan[1],
    )

    var precise_kernel_observation = List[Float32]()
    precise_kernel_observation.extend(pair(0.9, 0.4))
    precise_kernel_observation.extend(pair(0.1, 0.6))
    var precise_kernel_conditional = List[Float32]()
    precise_kernel_conditional.extend(pair(0.8, 0.3))
    precise_kernel_conditional.extend(pair(0.2, 0.7))
    var precise_obs_kernel = compute_precise_obs_kernels(
        log_values(precise_kernel_observation),
        log_values(precise_kernel_conditional),
        log_values(pair(0.6, 0.4)),
        0,
        1,
        2,
        1,
        2,
    )
    print(
        "precise_obs_kernel",
        precise_obs_kernel[0],
        precise_obs_kernel[1],
        precise_obs_kernel[2],
        precise_obs_kernel[3],
    )
    var precise_plan = precise_info_seeking_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
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
    print(
        "precise_plan",
        precise_plan[0],
        precise_plan[1],
        precise_plan[2],
        precise_plan[3],
        precise_plan[4],
        precise_plan[5],
        precise_plan[6],
        precise_plan[7],
        precise_plan[8],
        precise_plan[9],
        precise_plan[10],
        precise_plan[11],
        precise_plan[12],
        precise_plan[13],
        precise_plan[14],
        precise_plan[15],
        precise_plan[16],
        precise_plan[17],
        precise_plan[18],
        precise_plan[19],
        precise_plan[20],
        precise_plan[21],
        precise_plan[22],
        precise_plan[23],
        precise_plan[24],
        precise_plan[25],
        precise_plan[26],
        precise_plan[27],
        precise_plan[28],
        precise_plan[29],
        precise_plan[30],
        precise_plan[31],
        precise_plan[32],
        precise_plan[33],
    )
    var precise_theta_goal = List[Float32]()
    precise_theta_goal.extend(pair(0.15, 0.75))
    precise_theta_goal.extend(pair(0.85, 0.25))
    var precise_theta_plan = precise_info_seeking_planning_dense_theta_goal(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
        precise_theta_goal,
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
    print(
        "precise_theta_summary",
        precise_theta_plan[0],
        precise_theta_plan[1],
        precise_theta_plan[2],
        precise_theta_plan[3],
        precise_theta_plan[4],
        precise_theta_plan[5],
        precise_theta_plan[6],
        precise_theta_plan[7],
        precise_theta_plan[8],
        precise_theta_plan[9],
        precise_theta_plan[10],
        precise_theta_plan[33],
    )
    var precise_masked_plan = precise_info_seeking_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "precise_masked_summary",
        precise_masked_plan[0],
        precise_masked_plan[1],
        precise_masked_plan[2],
        precise_masked_plan[3],
        precise_masked_plan[4],
        precise_masked_plan[5],
        precise_masked_plan[6],
        precise_masked_plan[7],
        precise_masked_plan[8],
        precise_masked_plan[9],
    )

    var region_plan = region_extended_loopy_bp_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
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
    print(
        "region_plan",
        region_plan[0],
        region_plan[1],
        region_plan[2],
        region_plan[3],
        region_plan[4],
        region_plan[5],
        region_plan[6],
        region_plan[7],
        region_plan[8],
        region_plan[9],
        region_plan[10],
        region_plan[11],
        region_plan[12],
        region_plan[13],
        region_plan[14],
        region_plan[15],
        region_plan[16],
        region_plan[17],
        region_plan[18],
        region_plan[19],
        region_plan[20],
        region_plan[21],
        region_plan[22],
        region_plan[23],
        region_plan[24],
        region_plan[25],
        region_plan[26],
        region_plan[27],
        region_plan[28],
        region_plan[29],
        region_plan[30],
        region_plan[31],
        region_plan[32],
        region_plan[33],
        region_plan[34],
        region_plan[35],
        region_plan[36],
        region_plan[37],
        region_plan[38],
        region_plan[39],
        region_plan[40],
        region_plan[41],
    )
    var region_theta_plan = region_extended_loopy_bp_planning_dense_theta_goal(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
        precise_theta_goal,
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
    print(
        "region_theta_summary",
        region_theta_plan[0],
        region_theta_plan[1],
        region_theta_plan[2],
        region_theta_plan[17],
        region_theta_plan[18],
        region_theta_plan[41],
    )
    var region_masked_plan = region_extended_loopy_bp_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
    )
    print(
        "region_masked_summary",
        region_masked_plan[0],
        region_masked_plan[1],
        region_masked_plan[3],
        region_masked_plan[5],
    )

    var active_log_prior = log_values(pair(0.55, 0.45))
    var active_obs_local = precompute_obs_channels(
        log_values(dyn_plan_observation),
        active_log_prior,
        2,
        0.5,
        2,
        2,
        1,
        2,
    )
    var active_pref_local = precompute_pref_to_x(
        log_values(precise_theta_goal), active_log_prior, 2, 2
    )
    print(
        "active_precomputed_messages",
        active_obs_local[0],
        active_obs_local[1],
        active_pref_local[0],
        active_pref_local[1],
    )
    var active_dense_plan = active_inference_planning_dense(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
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
    print(
        "active_dense_plan",
        active_dense_plan[0],
        active_dense_plan[1],
        active_dense_plan[2],
        active_dense_plan[3],
        active_dense_plan[4],
        active_dense_plan[5],
        active_dense_plan[6],
        active_dense_plan[7],
        active_dense_plan[8],
        active_dense_plan[9],
        active_dense_plan[10],
        active_dense_plan[11],
        active_dense_plan[12],
        active_dense_plan[13],
        active_dense_plan[14],
        active_dense_plan[15],
        active_dense_plan[16],
        active_dense_plan[17],
        active_dense_plan[18],
        active_dense_plan[19],
        active_dense_plan[20],
        active_dense_plan[21],
        active_dense_plan[22],
        active_dense_plan[23],
        active_dense_plan[24],
        active_dense_plan[25],
        active_dense_plan[26],
        active_dense_plan[27],
        active_dense_plan[28],
        active_dense_plan[29],
        active_dense_plan[30],
        active_dense_plan[31],
        active_dense_plan[32],
        active_dense_plan[33],
        active_dense_plan[34],
        active_dense_plan[35],
        active_dense_plan[36],
        active_dense_plan[37],
        active_dense_plan[38],
        active_dense_plan[39],
        active_dense_plan[40],
        active_dense_plan[41],
    )
    var active_dense_theta_plan = active_inference_planning_dense_theta_goal(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        loopy_vbp_transition(),
        dyn_plan_observation,
        precise_theta_goal,
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
    print(
        "active_dense_theta_summary",
        active_dense_theta_plan[0],
        active_dense_theta_plan[1],
        active_dense_theta_plan[2],
        active_dense_theta_plan[17],
        active_dense_theta_plan[18],
        active_dense_theta_plan[41],
    )
    var active_log_base = _compute_dense_log_base(
        log_values(loopy_vbp_transition()), active_log_prior, 2, 2, 2
    )
    var active_precomputed_plan = active_inference_planning_precomputed(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        active_log_base,
        active_obs_local,
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    print(
        "active_precomputed_plan",
        active_precomputed_plan[0],
        active_precomputed_plan[1],
        active_precomputed_plan[2],
        active_precomputed_plan[3],
        active_precomputed_plan[4],
        active_precomputed_plan[5],
        active_precomputed_plan[6],
        active_precomputed_plan[7],
        active_precomputed_plan[8],
        active_precomputed_plan[9],
        active_precomputed_plan[10],
        active_precomputed_plan[11],
        active_precomputed_plan[12],
        active_precomputed_plan[13],
        active_precomputed_plan[14],
        active_precomputed_plan[15],
        active_precomputed_plan[16],
        active_precomputed_plan[17],
    )
    var active_theta_local = active_obs_local.copy()
    active_theta_local[0] += active_pref_local[0]
    active_theta_local[1] += active_pref_local[1]
    var active_precomputed_theta = (
        active_inference_planning_precomputed_theta_goal(
            pair(0.65, 0.35),
            pair(0.55, 0.45),
            active_log_base,
            active_theta_local,
            precise_theta_goal,
            pair(0.6, 0.4),
            2,
            2,
            0.5,
            2,
            2,
            2,
        )
    )
    print(
        "active_precomputed_theta_summary",
        active_precomputed_theta[0],
        active_precomputed_theta[1],
        active_precomputed_theta[2],
        active_precomputed_theta[17],
    )
    var active_sparse_base = compute_log_base_sparse(
        two_state_transition_indices(), active_log_prior, 2, 2, 2
    )
    var active_sparse_plan = active_inference_planning_precomputed(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        active_sparse_base,
        active_obs_local,
        pair(0.15, 0.85),
        pair(0.6, 0.4),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    print(
        "active_sparse_plan",
        active_sparse_plan[0],
        active_sparse_plan[1],
        active_sparse_plan[2],
        active_sparse_plan[3],
        active_sparse_plan[4],
        active_sparse_plan[5],
        active_sparse_plan[6],
        active_sparse_plan[7],
        active_sparse_plan[8],
        active_sparse_plan[9],
        active_sparse_plan[10],
        active_sparse_plan[11],
        active_sparse_plan[12],
        active_sparse_plan[13],
        active_sparse_plan[14],
        active_sparse_plan[15],
        active_sparse_plan[16],
        active_sparse_plan[17],
    )
    var active_masked_plan = active_inference_planning_precomputed(
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        active_sparse_base,
        active_obs_local,
        pair(0.15, 0.85),
        pair(1.0, 0.0),
        2,
        2,
        0.5,
        2,
        2,
        2,
    )
    print(
        "active_masked_summary",
        active_masked_plan[0],
        active_masked_plan[1],
        active_masked_plan[2],
        active_masked_plan[17],
    )
    var active_kernel_transition = List[Float32]()
    active_kernel_transition.extend(pair(0.8, 0.0))
    active_kernel_transition.extend(pair(0.4, 0.6))
    active_kernel_transition.extend(pair(0.2, 1.0))
    active_kernel_transition.extend(pair(0.6, 0.4))
    active_kernel_transition.extend(pair(0.3, 0.7))
    active_kernel_transition.extend(pair(0.0, 0.5))
    active_kernel_transition.extend(pair(0.7, 0.3))
    active_kernel_transition.extend(pair(1.0, 0.5))
    var active_kernel_actions = List[Float32]()
    active_kernel_actions.extend(pair(0.75, 0.25))
    active_kernel_actions.extend(pair(0.4, 0.6))
    active_kernel_actions.extend(pair(0.2, 0.8))
    active_kernel_actions.extend(pair(0.9, 0.1))
    var active_kernel_dyn = List[Float32]()
    active_kernel_dyn.extend(pair(0.6, 0.0))
    active_kernel_dyn.extend(pair(0.4, 1.0))
    active_kernel_dyn.extend(pair(0.3, 0.7))
    active_kernel_dyn.extend(pair(0.7, 0.3))
    active_kernel_dyn.extend(pair(0.1, 0.8))
    active_kernel_dyn.extend(pair(0.9, 0.2))
    active_kernel_dyn.extend(pair(0.0, 0.4))
    active_kernel_dyn.extend(pair(1.0, 0.6))
    var active_kernel = compute_dyn_kernels_aif(
        log_values(active_kernel_transition),
        log_values(active_kernel_actions),
        log_values(active_kernel_dyn),
        2,
        2,
        2,
        2,
    )
    for value in active_kernel:
        print("active_dyn_kernel", value)

    var temporal = flip_or_stay_reduced(2)
    var action_prior = pair(safe_log(0.7), safe_log(0.3))
    var fwd = forward_pass(
        temporal, pair(safe_log(0.8), safe_log(0.2)), action_prior, 2, 2, 2
    )
    print("loopy_fwd", fwd[0], fwd[1], fwd[2], fwd[3], fwd[4], fwd[5])
    var bwd = backward_messages(
        temporal, pair(safe_log(0.1), safe_log(0.9)), action_prior, 2, 2, 2
    )
    print("loopy_bwd", bwd[0], bwd[1], bwd[2], bwd[3], bwd[4], bwd[5])
    var action_marginals = compute_action_marginals(
        temporal, fwd, bwd, action_prior, 2, 2, 2
    )
    print(
        "loopy_actions",
        action_marginals[0],
        action_marginals[1],
        action_marginals[2],
        action_marginals[3],
    )
    var dyn_messages = List[Float32]()
    dyn_messages.extend(pair(safe_log(0.9), safe_log(0.2)))
    dyn_messages.extend(pair(safe_log(0.3), safe_log(0.8)))
    var theta_cavity = compute_theta_cavities(
        pair(safe_log(0.6), safe_log(0.4)), dyn_messages, 2, 2
    )
    print(
        "theta_cavity",
        theta_cavity[0],
        theta_cavity[1],
        theta_cavity[2],
        theta_cavity[3],
    )

    var planner_transitions = List[Int]()
    planner_transitions.append(0)
    planner_transitions.append(1)
    planner_transitions.append(1)
    planner_transitions.append(0)
    planner_transitions.append(1)
    planner_transitions.append(0)
    planner_transitions.append(0)
    planner_transitions.append(1)
    var plan = loopy_bp_planning_sparse(
        pair(1.0, 0.0),
        pair(0.9, 0.1),
        planner_transitions,
        pair(0.01, 0.99),
        pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    print("loopy_plan", plan[0], plan[1])
    var theta_goal = List[Float32]()
    theta_goal.append(0.01)
    theta_goal.append(0.9)
    theta_goal.append(0.99)
    theta_goal.append(0.1)
    var theta_plan = loopy_bp_planning_sparse_theta_goal(
        pair(1.0, 0.0),
        pair(0.9, 0.1),
        planner_transitions,
        theta_goal,
        pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    print("loopy_theta_plan", theta_plan[0], theta_plan[1])
    var dense_plan = loopy_bp_planning_dense(
        pair(1.0, 0.0),
        pair(0.9, 0.1),
        dense_uncertain_transition(),
        pair(0.01, 0.99),
        pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    print("loopy_dense_plan", dense_plan[0], dense_plan[1])
    var dense_theta_plan = loopy_bp_planning_dense_theta_goal(
        pair(1.0, 0.0),
        pair(0.9, 0.1),
        dense_uncertain_transition(),
        theta_goal,
        pair(0.5, 0.5),
        1,
        3,
        2,
        2,
        2,
    )
    print("loopy_dense_theta_plan", dense_theta_plan[0], dense_theta_plan[1])
    var masked_plan = loopy_bp_planning_sparse(
        pair(1.0, 0.0),
        pair(0.9, 0.1),
        planner_transitions,
        pair(0.01, 0.99),
        pair(1.0, 0.0),
        1,
        2,
        2,
        2,
        2,
    )
    print("loopy_masked_plan", masked_plan[0], masked_plan[1])

    var state_transitions = List[Int]()
    state_transitions.append(0)
    state_transitions.append(1)
    state_transitions.append(1)
    state_transitions.append(1)
    var observations = List[Float32]()
    observations.extend(pair(0.9, 0.9))
    observations.extend(pair(0.2, 0.2))
    observations.extend(pair(0.1, 0.1))
    observations.extend(pair(0.8, 0.8))
    var orientations = List[Float32]()
    orientations.extend(pair(0.8, 0.3))
    orientations.extend(pair(0.2, 0.7))
    var one_action = List[Float32]()
    one_action.append(1.0)
    var inferred = state_inference_step_sparse(
        pair(0.6, 0.4),
        pair(0.5, 0.5),
        state_transitions,
        observations,
        orientations,
        pair(1.0, 0.0),
        pair(1.0, 0.0),
        one_action,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
    )
    print("state_inference", inferred[0], inferred[1], inferred[2], inferred[3])
    var dense_state_transitions = List[Float32]()
    dense_state_transitions.extend(pair(1.0, 0.0))
    dense_state_transitions.extend(pair(0.0, 0.0))
    dense_state_transitions.extend(pair(0.0, 1.0))
    dense_state_transitions.extend(pair(1.0, 1.0))
    var dense_inferred = state_inference_step(
        pair(0.6, 0.4),
        pair(0.5, 0.5),
        dense_state_transitions,
        observations,
        orientations,
        pair(1.0, 0.0),
        pair(1.0, 0.0),
        one_action,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
    )
    print(
        "state_inference_dense",
        dense_inferred[0],
        dense_inferred[1],
        dense_inferred[2],
        dense_inferred[3],
    )

    var entropy_value = entropy(pair(safe_log(0.25), safe_log(0.75)))
    var entropy_unnorm_value = entropy_unnormalized(
        pair(safe_log(2.0), safe_log(6.0))
    )
    var energy_value = energy(
        pair(safe_log(0.25), safe_log(0.75)),
        pair(safe_log(0.8), safe_log(0.2)),
    )
    var structural_energy = energy(
        pair(safe_log(0.25), safe_log(0.75)),
        pair(safe_log(0.8), safe_log(0.0)),
    )
    print(
        "convergence_helpers",
        entropy_value,
        entropy_unnorm_value,
        energy_value,
        structural_energy,
    )
