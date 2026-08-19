"""Execute identical message fixtures in JAX and Mojo and compare outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "UAI-MP-AIF-JAX"))

from inference.messages import (  # noqa: E402
    combine_messages,
    combine_messages_log,
    backward_message_2d,
    backward_message_3d,
    backward_message_to_other_3d,
    forward_message_2d,
    forward_message_3d,
    forward_message_4d,
    marginalize_static,
    safe_log,
    safe_log_div,
    compute_log_base_sparse,
    sparse_reduced,
    sparse_reduced_weighted,
    sparse_dyn_to_theta,
    sparse_dyn_to_theta_weighted,
    sparse_dyn_to_theta_dyn_channel,
    sparse_dyn_channels_and_pair,
    sparse_dyn_channels_and_pair_dyn_channel,
    sparse_efe_action_prior,
    sparse_pair_marginal,
    sparse_reduced_dyn_channel,
)
from inference.active_inference import (  # noqa: E402
    active_inference_planning,
    compute_dyn_kernels_aif,
    precompute_obs_channels,
    precompute_pref_to_x,
)
from inference.loopy_bp import (  # noqa: E402
    backward_pass,
    compute_reduced_per_t,
    compute_theta_cavities,
    forward_pass,
    loopy_bp_planning,
)
from inference.loopy_vbp import (  # noqa: E402
    backward_pass_vbp,
    compute_dyn_to_theta_msgs_vbp,
    forward_pass_vbp,
    loopy_vbp_planning,
)
from inference.state_inference import (  # noqa: E402
    state_inference_step,
    state_inference_step_sparse,
)
from inference.convergence import _energy, _entropy, _entropy_unnorm  # noqa: E402
from inference.dyn_channel_loopy_bp import (  # noqa: E402
    dyn_channel_loopy_bp_planning,
)
from inference.vbp_channel import vbp_channel_planning  # noqa: E402
from inference.nuijten_mp import nuijten_mp_planning  # noqa: E402
from inference.precise_info_seeking import (  # noqa: E402
    precise_info_seeking_planning,
)
from inference.region_extended_loopy_bp import (  # noqa: E402
    region_extended_loopy_bp_planning,
)


def jax_outputs() -> dict[str, np.ndarray]:
    tensor2 = np.array([[0.8, 0.1], [0.2, 0.9]], dtype=np.float32)
    xor3 = np.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        dtype=np.float32,
    )
    parity4 = np.array(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
        ],
        dtype=np.float32,
    )
    log_tensor = safe_log(
        np.array([[[[0.9], [0.2]]], [[[0.1], [0.8]]]], dtype=np.float32)
    )
    transition_indices = jnp.array([[[0, 1]], [[1, 1]]], dtype=jnp.int32)
    dense_dyn_transition = jnp.array(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=jnp.float32,
    )
    log_cavities = safe_log(
        np.array([[0.25, 0.75], [0.6, 0.4]], dtype=np.float32)
    )
    temporal = safe_log(
        jnp.array(
            [
                [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
                [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=jnp.float32,
        )
    )
    action_prior = safe_log(jnp.array([0.7, 0.3], dtype=jnp.float32))
    sparse_fwd = safe_log(jnp.array([[0.6, 0.4], [0.5, 0.5]], dtype=jnp.float32))
    sparse_bwd = safe_log(jnp.array([[0.5, 0.5], [0.7, 0.3]], dtype=jnp.float32))
    sparse_local = jnp.zeros((2, 2), dtype=jnp.float32)
    sparse_action = jnp.zeros((1,), dtype=jnp.float32)
    sparse_kernel_weight = safe_log(
        jnp.array([[[0.5], [0.2]]], dtype=jnp.float32)
    )
    sparse_dyn_channels = safe_log(
        jnp.array([[[[0.4], [0.6]], [[0.2], [0.8]]]], dtype=jnp.float32)
    )
    sparse_action_channels = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    loopy_fwd = forward_pass(
        temporal,
        safe_log(jnp.array([0.8, 0.2], dtype=jnp.float32)),
        action_prior,
        2,
    )
    loopy_bwd, loopy_actions = backward_pass(
        temporal,
        loopy_fwd,
        safe_log(jnp.array([0.1, 0.9], dtype=jnp.float32)),
        action_prior,
        2,
    )
    nuijten_args = (
        jnp.array([0.8, 0.2], dtype=jnp.float32),
        jnp.array([0.6, 0.4], dtype=jnp.float32),
        dense_dyn_transition,
        jnp.array(
            [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
            dtype=jnp.float32,
        ),
    )
    nuijten_action_prior = jnp.array([0.55, 0.45], dtype=jnp.float32)
    nuijten_indices = jnp.array(
        [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
    )
    nuijten_sparse_action, _, nuijten_sparse_obs = nuijten_mp_planning(
        *nuijten_args,
        jnp.array([0.1, 0.9], dtype=jnp.float32),
        2,
        2,
        action_prior=nuijten_action_prior,
        T_idx=nuijten_indices,
    )
    nuijten_sparse_theta_action, _, nuijten_sparse_theta_obs = (
        nuijten_mp_planning(
            *nuijten_args,
            jnp.array([[0.1, 0.8], [0.9, 0.2]], dtype=jnp.float32),
            2,
            2,
            action_prior=nuijten_action_prior,
            T_idx=nuijten_indices,
        )
    )
    nuijten_dense_action, nuijten_dense_dyn, nuijten_dense_obs = (
        nuijten_mp_planning(
            *nuijten_args,
            jnp.array([0.1, 0.9], dtype=jnp.float32),
            2,
            2,
            action_prior=nuijten_action_prior,
        )
    )
    nuijten_dense_theta_action, nuijten_dense_theta_dyn, nuijten_dense_theta_obs = (
        nuijten_mp_planning(
            *nuijten_args,
            jnp.array([[0.1, 0.8], [0.9, 0.2]], dtype=jnp.float32),
            2,
            2,
            action_prior=nuijten_action_prior,
        )
    )
    loopy_vbp_transition = jnp.array(
        [
            [
                [[0.9, 0.2], [0.4, 0.7]],
                [[0.3, 0.8], [0.6, 0.1]],
            ],
            [
                [[0.1, 0.8], [0.6, 0.3]],
                [[0.7, 0.2], [0.4, 0.9]],
            ],
        ],
        dtype=jnp.float32,
    )
    loopy_vbp_q0 = jnp.array([0.65, 0.35], dtype=jnp.float32)
    loopy_vbp_theta = jnp.array([0.55, 0.45], dtype=jnp.float32)
    loopy_vbp_log_transition = safe_log(loopy_vbp_transition)
    loopy_vbp_reduced = compute_reduced_per_t(
        loopy_vbp_log_transition,
        jnp.tile(safe_log(loopy_vbp_theta), (2, 1)),
    )
    loopy_vbp_values, loopy_vbp_q_values = backward_pass_vbp(
        loopy_vbp_reduced,
        safe_log(jnp.array([0.15, 0.85], dtype=jnp.float32)),
        2,
    )
    loopy_vbp_forward = forward_pass_vbp(
        loopy_vbp_reduced,
        safe_log(loopy_vbp_q0),
        loopy_vbp_q_values,
        2,
    )
    loopy_vbp_dyn_theta = compute_dyn_to_theta_msgs_vbp(
        loopy_vbp_log_transition,
        loopy_vbp_forward,
        loopy_vbp_values,
        2,
    )
    loopy_vbp_plan = loopy_vbp_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
    )
    loopy_vbp_theta_plan = loopy_vbp_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        jnp.array([[0.15, 0.75], [0.85, 0.25]], dtype=jnp.float32),
        2,
        2,
    )
    precise_observation = nuijten_args[3]
    precise_kernel_observation = jnp.array(
        [[[[0.9, 0.4]], [[0.1, 0.6]]]], dtype=jnp.float32
    )
    precise_kernel_conditional = jnp.array(
        [[[[[0.8, 0.3]], [[0.2, 0.7]]]]], dtype=jnp.float32
    )
    precise_kernel_marginal = jnp.array(
        [[[[0.6], [0.4]]]], dtype=jnp.float32
    )
    precise_obs_kernel = (
        safe_log(precise_kernel_observation)[None]
        + safe_log(precise_kernel_conditional)
        + safe_log(precise_kernel_conditional)
        - safe_log(precise_kernel_marginal)[..., None]
    )
    precise_plan = precise_info_seeking_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
    )
    precise_theta_plan = precise_info_seeking_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([[0.15, 0.75], [0.85, 0.25]], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
    )
    precise_masked_plan = precise_info_seeking_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    region_plan = region_extended_loopy_bp_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
    )
    region_theta_plan = region_extended_loopy_bp_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([[0.15, 0.75], [0.85, 0.25]], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
    )
    region_masked_plan = region_extended_loopy_bp_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    active_log_prior = safe_log(loopy_vbp_theta)
    active_obs_local = precompute_obs_channels(
        safe_log(precise_observation), active_log_prior, 2, damping=0.5
    )
    active_pref_local = precompute_pref_to_x(
        safe_log(jnp.array([[0.15, 0.75], [0.85, 0.25]], dtype=jnp.float32)),
        active_log_prior,
    )
    active_log_base = jax.scipy.special.logsumexp(
        safe_log(loopy_vbp_transition).transpose(1, 0, 2, 3)
        + active_log_prior[None, None, :, None],
        axis=2,
    )
    active_dense_plan = active_inference_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
    )
    active_dense_theta_plan = active_inference_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        loopy_vbp_transition,
        precise_observation,
        jnp.array([[0.15, 0.75], [0.85, 0.25]], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
    )
    active_precomputed_plan = active_inference_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        None,
        None,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
        log_base=active_log_base,
        log_local_to_x=active_obs_local,
    )
    active_precomputed_theta = active_inference_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        None,
        None,
        jnp.array([[0.15, 0.75], [0.85, 0.25]], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
        log_base=active_log_base,
        log_local_to_x=active_obs_local + active_pref_local,
    )
    active_sparse_base = compute_log_base_sparse(
        nuijten_indices, active_log_prior, 2
    )
    active_sparse_plan = active_inference_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        None,
        None,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([0.6, 0.4], dtype=jnp.float32),
        log_base=active_sparse_base,
        log_local_to_x=active_obs_local,
    )
    active_masked_plan = active_inference_planning(
        loopy_vbp_q0,
        loopy_vbp_theta,
        None,
        None,
        jnp.array([0.15, 0.85], dtype=jnp.float32),
        2,
        2,
        damping=0.5,
        action_prior=jnp.array([1.0, 0.0], dtype=jnp.float32),
        log_base=active_sparse_base,
        log_local_to_x=active_obs_local,
    )
    active_kernel_transition = jnp.array(
        [
            0.8, 0.0, 0.4, 0.6, 0.2, 1.0, 0.6, 0.4,
            0.3, 0.7, 0.0, 0.5, 0.7, 0.3, 1.0, 0.5,
        ],
        dtype=jnp.float32,
    ).reshape(2, 2, 2, 2)
    active_kernel_actions = jnp.array(
        [0.75, 0.25, 0.4, 0.6, 0.2, 0.8, 0.9, 0.1], dtype=jnp.float32
    ).reshape(2, 2, 2)
    active_kernel_dyn = jnp.array(
        [
            0.6, 0.0, 0.4, 1.0, 0.3, 0.7, 0.7, 0.3,
            0.1, 0.8, 0.9, 0.2, 0.0, 0.4, 1.0, 0.6,
        ],
        dtype=jnp.float32,
    ).reshape(2, 2, 2, 2)
    active_dyn_kernel = compute_dyn_kernels_aif(
        safe_log(active_kernel_transition),
        safe_log(active_kernel_actions),
        safe_log(active_kernel_dyn),
    )
    return {
        "softmax_impossible": np.asarray(
            jax.nn.softmax(jnp.array([-1.0e12, -1.0e12], dtype=jnp.float32))
        ),
        "forward2d": np.asarray(
            forward_message_2d(tensor2, np.array([0.25, 0.75], dtype=np.float32))
        ),
        "backward2d": np.asarray(
            backward_message_2d(tensor2, np.array([1.0, 0.0], dtype=np.float32))
        ),
        "forward3d": np.asarray(
            forward_message_3d(
                xor3,
                np.array([0.25, 0.75], dtype=np.float32),
                np.array([0.6, 0.4], dtype=np.float32),
            )
        ),
        "backward3d": np.asarray(
            backward_message_3d(
                xor3,
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.6, 0.4], dtype=np.float32),
            )
        ),
        "backward_other3d": np.asarray(
            backward_message_to_other_3d(
                xor3,
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.25, 0.75], dtype=np.float32),
            )
        ),
        "forward4d": np.asarray(
            forward_message_4d(
                parity4,
                np.array([0.2, 0.8], dtype=np.float32),
                np.array([0.3, 0.7], dtype=np.float32),
                np.array([0.4, 0.6], dtype=np.float32),
            )
        ),
        "combined": np.asarray(
            combine_messages(
                [
                    np.array([0.8, 0.2], dtype=np.float32),
                    np.array([0.6, 0.4], dtype=np.float32),
                ]
            )
        ),
        "combined_log": np.asarray(
            combine_messages_log(
                [
                    safe_log(np.array([0.8, 0.2], dtype=np.float32)),
                    safe_log(np.array([0.6, 0.4], dtype=np.float32)),
                ]
            )
        ),
        "safe_log_edges": np.asarray(
            [
                safe_log(jnp.float32(0.0)),
                safe_log(jnp.float32(1.0e-30)),
                safe_log(jnp.float32(0.5)),
                safe_log(jnp.float32(-1.0)),
                safe_log_div(safe_log(jnp.float32(0.5)), safe_log(jnp.float32(0.25))),
                safe_log_div(safe_log(jnp.float32(0.0)), safe_log(jnp.float32(0.25))),
                safe_log_div(safe_log(jnp.float32(0.5)), safe_log(jnp.float32(0.0))),
                safe_log_div(safe_log(jnp.float32(0.0)), safe_log(jnp.float32(0.0))),
            ],
            dtype=np.float32,
        ),
        "marginalized": np.asarray(
            marginalize_static(
                log_tensor, safe_log(np.array([0.25, 0.75], dtype=np.float32))
            )
        ).reshape(-1),
        "sparse_base": np.asarray(
            compute_log_base_sparse(
                transition_indices,
                safe_log(np.array([0.25, 0.75], dtype=np.float32)),
                2,
            )
        ).reshape(-1),
        "sparse_reduced": np.asarray(
            sparse_reduced(transition_indices, log_cavities, 2)
        ).reshape(-1),
        "sparse_weighted": np.asarray(
            sparse_reduced_weighted(
                transition_indices,
                log_cavities[:1],
                safe_log(np.array([[[0.5], [0.2]]], dtype=np.float32)),
                2,
            )
        ).reshape(-1),
        "sparse_dyn_channel_reduced": np.asarray(
            sparse_reduced_dyn_channel(
                transition_indices,
                safe_log(jnp.array([[0.25, 0.75]], dtype=jnp.float32)),
                safe_log(
                    jnp.array([[[[0.4], [0.6]], [[0.2], [0.8]]]], dtype=jnp.float32)
                ),
                jnp.zeros((1, 2, 1), dtype=jnp.float32),
                2,
            )
        ).reshape(-1),
        "sparse_theta": np.asarray(
            sparse_dyn_to_theta(
                transition_indices,
                sparse_fwd,
                sparse_bwd,
                sparse_local,
                sparse_action,
                2,
            )
        ).reshape(-1),
        "sparse_theta_weighted": np.asarray(
            sparse_dyn_to_theta_weighted(
                transition_indices,
                sparse_fwd,
                sparse_bwd,
                sparse_local,
                sparse_action,
                sparse_kernel_weight,
                2,
            )
        ).reshape(-1),
        "sparse_pair": np.asarray(
            sparse_pair_marginal(
                transition_indices,
                sparse_fwd,
                sparse_bwd,
                sparse_local,
                safe_log(jnp.array([[0.25, 0.75]], dtype=jnp.float32)),
                sparse_action,
                None,
                2,
            )
        ).reshape(-1),
        "sparse_pair_weighted": np.asarray(
            sparse_pair_marginal(
                transition_indices,
                sparse_fwd,
                sparse_bwd,
                sparse_local,
                safe_log(jnp.array([[0.25, 0.75]], dtype=jnp.float32)),
                sparse_action,
                sparse_kernel_weight,
                2,
            )
        ).reshape(-1),
        "sparse_dyn_channel_theta": np.asarray(
            sparse_dyn_to_theta_dyn_channel(
                transition_indices,
                sparse_fwd,
                sparse_bwd,
                sparse_local,
                sparse_action,
                sparse_dyn_channels,
                sparse_action_channels,
                2,
            )
        ).reshape(-1),
        "sparse_dyn_channel_pair": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in sparse_dyn_channels_and_pair_dyn_channel(
                    transition_indices,
                    sparse_fwd,
                    sparse_bwd,
                    sparse_local,
                    safe_log(jnp.array([[0.25, 0.75]], dtype=jnp.float32)),
                    sparse_action,
                    sparse_dyn_channels,
                    sparse_action_channels,
                    2,
                )
            )
        ),
        "sparse_channel_pair": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in sparse_dyn_channels_and_pair(
                    transition_indices,
                    sparse_fwd,
                    sparse_bwd,
                    sparse_local,
                    safe_log(jnp.array([[0.25, 0.75]], dtype=jnp.float32)),
                    sparse_action,
                    None,
                    2,
                )
            )
        ),
        "sparse_weighted_channel_pair": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in sparse_dyn_channels_and_pair(
                    transition_indices,
                    sparse_fwd,
                    sparse_bwd,
                    sparse_local,
                    safe_log(jnp.array([[0.25, 0.75]], dtype=jnp.float32)),
                    sparse_action,
                    sparse_kernel_weight,
                    2,
                )
            )
        ),
        "sparse_efe": np.asarray(
            sparse_efe_action_prior(
                jnp.array(
                    [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                ),
                sparse_fwd,
                sparse_bwd,
                sparse_local,
                safe_log(jnp.array([[0.7, 0.3]], dtype=jnp.float32)),
                safe_log(jnp.array([[0.5, 0.5]], dtype=jnp.float32)),
                2,
                jnp.array([1.0, 1.0], dtype=jnp.float32),
            )
        ).reshape(-1),
        "sparse_efe_masked": np.asarray(
            sparse_efe_action_prior(
                jnp.array(
                    [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                ),
                sparse_fwd,
                sparse_bwd,
                sparse_local,
                safe_log(jnp.array([[0.7, 0.3]], dtype=jnp.float32)),
                safe_log(jnp.array([[0.5, 0.5]], dtype=jnp.float32)),
                2,
                jnp.array([1.0, 0.0], dtype=jnp.float32),
            )
        ).reshape(-1),
        "dyn_channel_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in dyn_channel_loopy_bp_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([0.1, 0.9], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                    T_idx=jnp.array(
                        [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                    ),
                )
            )
        ),
        "dyn_channel_theta_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in dyn_channel_loopy_bp_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([[0.1, 0.8], [0.9, 0.2]], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                    T_idx=jnp.array(
                        [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                    ),
                )
            )
        ),
        "dyn_channel_masked_plan": np.asarray(
            dyn_channel_loopy_bp_planning(
                jnp.array([0.8, 0.2], dtype=jnp.float32),
                jnp.array([0.6, 0.4], dtype=jnp.float32),
                jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
                jnp.array(
                    [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                    dtype=jnp.float32,
                ),
                jnp.array([0.1, 0.9], dtype=jnp.float32),
                2,
                2,
                damping=0.5,
                action_prior=jnp.array([1.0, 0.0], dtype=jnp.float32),
                T_idx=jnp.array(
                    [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                ),
            )[0]
        ).reshape(-1),
        "dyn_channel_dense_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in dyn_channel_loopy_bp_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    dense_dyn_transition,
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([0.1, 0.9], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                )
            )
        ),
        "dyn_channel_dense_theta_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in dyn_channel_loopy_bp_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    dense_dyn_transition,
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([[0.1, 0.8], [0.9, 0.2]], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                )
            )
        ),
        "vbp_sparse_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in vbp_channel_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    dense_dyn_transition,
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([0.1, 0.9], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                    T_idx=jnp.array(
                        [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                    ),
                )
            )
        ),
        "vbp_dense_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in vbp_channel_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    dense_dyn_transition,
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([0.1, 0.9], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                )
            )
        ),
        "vbp_sparse_theta_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in vbp_channel_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    dense_dyn_transition,
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([[0.1, 0.8], [0.9, 0.2]], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                    T_idx=jnp.array(
                        [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                    ),
                )
            )
        ),
        "vbp_dense_theta_plan": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in vbp_channel_planning(
                    jnp.array([0.8, 0.2], dtype=jnp.float32),
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    dense_dyn_transition,
                    jnp.array(
                        [[[[0.9, 0.2], [0.3, 0.8]], [[0.1, 0.8], [0.7, 0.2]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([[0.1, 0.8], [0.9, 0.2]], dtype=jnp.float32),
                    2,
                    2,
                    damping=0.5,
                    action_prior=jnp.array([0.55, 0.45], dtype=jnp.float32),
                )
            )
        ),
        "nuijten_sparse_plan": np.concatenate(
            (
                np.asarray(nuijten_sparse_action).reshape(-1),
                np.asarray(nuijten_sparse_obs).reshape(-1),
            )
        ),
        "nuijten_sparse_theta_plan": np.concatenate(
            (
                np.asarray(nuijten_sparse_theta_action).reshape(-1),
                np.asarray(nuijten_sparse_theta_obs).reshape(-1),
            )
        ),
        "nuijten_dense_summary": np.asarray(
            [
                nuijten_dense_action[0],
                nuijten_dense_action[1],
                nuijten_dense_dyn.reshape(-1)[0],
                nuijten_dense_dyn.reshape(-1)[-1],
                nuijten_dense_obs.reshape(-1)[0],
                nuijten_dense_obs.reshape(-1)[-1],
            ],
            dtype=np.float32,
        ),
        "nuijten_dense_theta_summary": np.asarray(
            [
                nuijten_dense_theta_action[0],
                nuijten_dense_theta_action[1],
                nuijten_dense_theta_dyn.reshape(-1)[0],
                nuijten_dense_theta_dyn.reshape(-1)[-1],
                nuijten_dense_theta_obs.reshape(-1)[0],
                nuijten_dense_theta_obs.reshape(-1)[-1],
            ],
            dtype=np.float32,
        ),
        "loopy_vbp_backward": np.concatenate(
            (
                np.asarray(loopy_vbp_values).reshape(-1),
                np.asarray(loopy_vbp_q_values).reshape(-1),
            )
        ),
        "loopy_vbp_forward": np.asarray(loopy_vbp_forward).reshape(-1),
        "loopy_vbp_dyn_theta": np.asarray(loopy_vbp_dyn_theta).reshape(-1),
        "loopy_vbp_plan": np.asarray(loopy_vbp_plan).reshape(-1),
        "loopy_vbp_theta_plan": np.asarray(loopy_vbp_theta_plan).reshape(-1),
        "precise_obs_kernel": np.asarray(precise_obs_kernel).reshape(-1),
        "precise_plan": np.concatenate(
            tuple(np.asarray(value).reshape(-1) for value in precise_plan)
        ),
        "precise_theta_summary": np.concatenate(
            (
                np.asarray(precise_theta_plan[0]).reshape(-1),
                np.asarray(precise_theta_plan[1]).reshape(-1),
                np.asarray(precise_theta_plan[2]).reshape(-1)[[0, -1]],
            )
        ),
        "precise_masked_summary": np.concatenate(
            (
                np.asarray(precise_masked_plan[0]).reshape(-1),
                np.asarray(precise_masked_plan[1]).reshape(-1),
            )
        ),
        "region_plan": np.concatenate(
            tuple(np.asarray(value).reshape(-1) for value in region_plan)
        ),
        "region_theta_summary": np.asarray(
            [
                region_theta_plan[0][0],
                region_theta_plan[0][1],
                region_theta_plan[1].reshape(-1)[0],
                region_theta_plan[1].reshape(-1)[-1],
                region_theta_plan[2].reshape(-1)[0],
                region_theta_plan[2].reshape(-1)[-1],
            ],
            dtype=np.float32,
        ),
        "region_masked_summary": np.asarray(
            [
                region_masked_plan[0][0],
                region_masked_plan[0][1],
                region_masked_plan[1].reshape(-1)[1],
                region_masked_plan[1].reshape(-1)[3],
            ],
            dtype=np.float32,
        ),
        "active_precomputed_messages": np.concatenate(
            (np.asarray(active_obs_local), np.asarray(active_pref_local))
        ),
        "active_dense_plan": np.concatenate(
            tuple(np.asarray(value).reshape(-1) for value in active_dense_plan)
        ),
        "active_dense_theta_summary": np.asarray(
            [
                active_dense_theta_plan[0][0],
                active_dense_theta_plan[0][1],
                active_dense_theta_plan[1].reshape(-1)[0],
                active_dense_theta_plan[1].reshape(-1)[-1],
                active_dense_theta_plan[2].reshape(-1)[0],
                active_dense_theta_plan[2].reshape(-1)[-1],
            ],
            dtype=np.float32,
        ),
        "active_precomputed_plan": np.concatenate(
            (
                np.asarray(active_precomputed_plan[0]).reshape(-1),
                np.asarray(active_precomputed_plan[1]).reshape(-1),
            )
        ),
        "active_precomputed_theta_summary": np.asarray(
            [
                active_precomputed_theta[0][0],
                active_precomputed_theta[0][1],
                active_precomputed_theta[1].reshape(-1)[0],
                active_precomputed_theta[1].reshape(-1)[-1],
            ],
            dtype=np.float32,
        ),
        "active_sparse_plan": np.concatenate(
            (
                np.asarray(active_sparse_plan[0]).reshape(-1),
                np.asarray(active_sparse_plan[1]).reshape(-1),
            )
        ),
        "active_masked_summary": np.asarray(
            [
                active_masked_plan[0][0],
                active_masked_plan[0][1],
                active_masked_plan[1].reshape(-1)[0],
                active_masked_plan[1].reshape(-1)[-1],
            ],
            dtype=np.float32,
        ),
        "active_dyn_kernel": np.asarray(active_dyn_kernel).reshape(-1),
        "loopy_fwd": np.asarray(loopy_fwd).reshape(-1),
        "loopy_bwd": np.asarray(loopy_bwd).reshape(-1),
        "loopy_actions": np.asarray(loopy_actions).reshape(-1),
        "theta_cavity": np.asarray(
            compute_theta_cavities(
                safe_log(jnp.array([0.6, 0.4], dtype=jnp.float32)),
                safe_log(jnp.array([[0.9, 0.2], [0.3, 0.8]], dtype=jnp.float32)),
            )
        ).reshape(-1),
        "loopy_plan": np.asarray(
            loopy_bp_planning(
                jnp.array([1.0, 0.0], dtype=jnp.float32),
                jnp.array([0.9, 0.1], dtype=jnp.float32),
                jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
                jnp.array([0.01, 0.99], dtype=jnp.float32),
                1,
                3,
                T_idx=jnp.array(
                    [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                ),
            )
        ).reshape(-1),
        "loopy_theta_plan": np.asarray(
            loopy_bp_planning(
                jnp.array([1.0, 0.0], dtype=jnp.float32),
                jnp.array([0.9, 0.1], dtype=jnp.float32),
                jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
                jnp.array([[0.01, 0.9], [0.99, 0.1]], dtype=jnp.float32),
                1,
                3,
                T_idx=jnp.array(
                    [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                ),
            )
        ).reshape(-1),
        "loopy_dense_plan": np.asarray(
            loopy_bp_planning(
                jnp.array([1.0, 0.0], dtype=jnp.float32),
                jnp.array([0.9, 0.1], dtype=jnp.float32),
                jnp.array(
                    [
                        [
                            [[1.0, 0.0], [0.0, 1.0]],
                            [[0.0, 1.0], [1.0, 0.0]],
                        ],
                        [
                            [[0.0, 1.0], [1.0, 0.0]],
                            [[1.0, 0.0], [0.0, 1.0]],
                        ],
                    ],
                    dtype=jnp.float32,
                ),
                jnp.array([0.01, 0.99], dtype=jnp.float32),
                1,
                3,
            )
        ).reshape(-1),
        "loopy_dense_theta_plan": np.asarray(
            loopy_bp_planning(
                jnp.array([1.0, 0.0], dtype=jnp.float32),
                jnp.array([0.9, 0.1], dtype=jnp.float32),
                jnp.array(
                    [
                        [
                            [[1.0, 0.0], [0.0, 1.0]],
                            [[0.0, 1.0], [1.0, 0.0]],
                        ],
                        [
                            [[0.0, 1.0], [1.0, 0.0]],
                            [[1.0, 0.0], [0.0, 1.0]],
                        ],
                    ],
                    dtype=jnp.float32,
                ),
                jnp.array([[0.01, 0.9], [0.99, 0.1]], dtype=jnp.float32),
                1,
                3,
            )
        ).reshape(-1),
        "loopy_masked_plan": np.asarray(
            loopy_bp_planning(
                jnp.array([1.0, 0.0], dtype=jnp.float32),
                jnp.array([0.9, 0.1], dtype=jnp.float32),
                jnp.zeros((2, 2, 2, 2), dtype=jnp.float32),
                jnp.array([0.01, 0.99], dtype=jnp.float32),
                1,
                2,
                action_prior=jnp.array([1.0, 0.0], dtype=jnp.float32),
                T_idx=jnp.array(
                    [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=jnp.int32
                ),
            )
        ).reshape(-1),
        "state_inference": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in state_inference_step_sparse(
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    jnp.array([0.5, 0.5], dtype=jnp.float32),
                    jnp.array([[[0, 1]], [[1, 1]]], dtype=jnp.int32),
                    jnp.array(
                        [[[[[0.9, 0.9], [0.2, 0.2]], [[0.1, 0.1], [0.8, 0.8]]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([[0.8, 0.3], [0.2, 0.7]], dtype=jnp.float32),
                    jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
                    jnp.array([1.0, 0.0], dtype=jnp.float32),
                    jnp.array([1.0], dtype=jnp.float32),
                    2,
                )
            )
        ),
        "state_inference_dense": np.concatenate(
            tuple(
                np.asarray(value).reshape(-1)
                for value in state_inference_step(
                    jnp.array([0.6, 0.4], dtype=jnp.float32),
                    jnp.array([0.5, 0.5], dtype=jnp.float32),
                    jnp.array(
                        [
                            [[[1.0], [0.0]], [[0.0], [0.0]]],
                            [[[0.0], [1.0]], [[1.0], [1.0]]],
                        ],
                        dtype=jnp.float32,
                    ),
                    jnp.array(
                        [[[[[0.9, 0.9], [0.2, 0.2]], [[0.1, 0.1], [0.8, 0.8]]]]],
                        dtype=jnp.float32,
                    ),
                    jnp.array([[0.8, 0.3], [0.2, 0.7]], dtype=jnp.float32),
                    jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
                    jnp.array([1.0, 0.0], dtype=jnp.float32),
                    jnp.array([1.0], dtype=jnp.float32),
                    2,
                )
            )
        ),
        "convergence_helpers": np.asarray(
            [
                _entropy(safe_log(jnp.array([0.25, 0.75], dtype=jnp.float32))),
                _entropy_unnorm(safe_log(jnp.array([2.0, 6.0], dtype=jnp.float32))),
                _energy(
                    safe_log(jnp.array([0.25, 0.75], dtype=jnp.float32)),
                    safe_log(jnp.array([0.8, 0.2], dtype=jnp.float32)),
                ),
                _energy(
                    safe_log(jnp.array([0.25, 0.75], dtype=jnp.float32)),
                    safe_log(jnp.array([0.8, 0.0], dtype=jnp.float32)),
                ),
            ],
            dtype=np.float32,
        ),
    }


def mojo_outputs() -> dict[str, np.ndarray]:
    completed = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            "src",
            "tests/mojo/parity_messages.mojo",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, list[float]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] in {
            "softmax_impossible",
            "forward2d",
            "backward2d",
            "forward3d",
            "backward3d",
            "backward_other3d",
            "forward4d",
            "combined",
            "combined_log",
            "safe_log_edges",
            "marginalized",
            "sparse_base",
            "sparse_reduced",
            "sparse_weighted",
            "sparse_dyn_channel_reduced",
            "sparse_theta",
            "sparse_theta_weighted",
            "sparse_pair",
            "sparse_pair_weighted",
            "sparse_dyn_channel_theta",
            "sparse_dyn_channel_pair",
            "sparse_channel_pair",
            "sparse_weighted_channel_pair",
            "sparse_efe",
            "sparse_efe_masked",
            "dyn_channel_plan",
            "dyn_channel_theta_plan",
            "dyn_channel_masked_plan",
            "dyn_channel_dense_plan",
            "dyn_channel_dense_theta_plan",
            "vbp_sparse_plan",
            "vbp_dense_plan",
            "vbp_sparse_theta_plan",
            "vbp_dense_theta_plan",
            "nuijten_sparse_plan",
            "nuijten_sparse_theta_plan",
            "nuijten_dense_summary",
            "nuijten_dense_theta_summary",
            "loopy_vbp_backward",
            "loopy_vbp_forward",
            "loopy_vbp_dyn_theta",
            "loopy_vbp_plan",
            "loopy_vbp_theta_plan",
            "precise_obs_kernel",
            "precise_plan",
            "precise_theta_summary",
            "precise_masked_summary",
            "region_plan",
            "region_theta_summary",
            "region_masked_summary",
            "active_precomputed_messages",
            "active_dense_plan",
            "active_dense_theta_summary",
            "active_precomputed_plan",
            "active_precomputed_theta_summary",
            "active_sparse_plan",
            "active_masked_summary",
            "active_dyn_kernel",
            "loopy_fwd",
            "loopy_bwd",
            "loopy_actions",
            "theta_cavity",
            "loopy_plan",
            "loopy_theta_plan",
            "loopy_dense_plan",
            "loopy_dense_theta_plan",
            "loopy_masked_plan",
            "state_inference",
            "state_inference_dense",
            "convergence_helpers",
        }:
            result.setdefault(parts[0], []).extend(float(value) for value in parts[1:])
    return {name: np.asarray(values, dtype=np.float32) for name, values in result.items()}


def main() -> None:
    expected = jax_outputs()
    actual = mojo_outputs()
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"Mojo output keys differ: expected {sorted(expected)}, got {sorted(actual)}"
        )
    for name, jax_value in expected.items():
        np.testing.assert_allclose(actual[name], jax_value, rtol=1e-5, atol=1e-6)
        print(f"PASS {name}: max_abs_error={np.max(np.abs(actual[name] - jax_value)):.3e}")


if __name__ == "__main__":
    main()
