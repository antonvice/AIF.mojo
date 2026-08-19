"""Compare complete VFE helper fixtures against the frozen JAX oracle."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "UAI-MP-AIF-JAX"))

from inference.convergence import (  # noqa: E402
    _action_cond_entropy,
    _dyn_cond_entropy,
    _obs_cond_entropy,
    active_inference_convergence,
    compute_active_inference_vfe,
    compute_bethe_vfe_loopy,
    compute_dyn_channel_vfe,
    compute_nuijten_vfe,
    compute_precise_info_seeking_vfe,
    compute_region_extended_vfe,
    compute_vbp_channel_vfe,
    dyn_channel_convergence,
    loopy_bp_convergence,
    nuijten_mp_convergence,
    precise_info_seeking_convergence,
    region_extended_convergence,
    vbp_channel_convergence,
)
from inference.messages import safe_log  # noqa: E402


def jax_values() -> np.ndarray:
    transition = jnp.array(
        [
            0.9, 0.2, 0.7, 0.4, 0.3, 0.8, 0.6, 0.1,
            0.1, 0.8, 0.3, 0.6, 0.7, 0.2, 0.4, 0.9,
        ],
        dtype=jnp.float32,
    ).reshape(2, 2, 2, 2)
    log_transition = safe_log(transition)
    reduced = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    fwd = safe_log(jnp.array([[0.8, 0.2], [0.55, 0.45], [0.4, 0.6]], dtype=jnp.float32))
    bwd = safe_log(jnp.array([[0.45, 0.55], [0.35, 0.65], [0.2, 0.8]], dtype=jnp.float32))
    q_u = jnp.array([[0.7, 0.3], [0.4, 0.6]], dtype=jnp.float32)
    dyn_cavity = safe_log(jnp.array([[0.6, 0.4], [0.3, 0.7]], dtype=jnp.float32))
    obs_cavity = safe_log(
        jnp.array([[0.5, 0.5], [0.8, 0.2], [0.25, 0.75]], dtype=jnp.float32)
    )
    theta_prior = safe_log(jnp.array([0.5, 0.5], dtype=jnp.float32))
    action_prior = safe_log(jnp.array([0.6, 0.4], dtype=jnp.float32))
    first_region = jnp.arange(1, 17, dtype=jnp.float32) / 136.0
    second_region = jnp.arange(17, 33, dtype=jnp.float32) / 392.0
    dyn_regions = safe_log(jnp.concatenate((first_region, second_region))).reshape(
        2, 2, 2, 2, 2
    )
    obs_probabilities = jnp.concatenate(
        tuple(
            jnp.arange(time * 8 + 1, time * 8 + 9, dtype=jnp.float32) / denominator
            for time, denominator in enumerate((36.0, 100.0, 164.0))
        )
    ).reshape(3, 1, 2, 2, 2)
    obs_regions = safe_log(obs_probabilities)
    obs_factor = safe_log(
        jnp.array([0.8, 0.3, 0.6, 0.4, 0.2, 0.7, 0.4, 0.6], dtype=jnp.float32)
    ).reshape(1, 2, 2, 2)
    tiled_transition = jnp.broadcast_to(
        log_transition.transpose(1, 0, 2, 3)[None], (2, 2, 2, 2, 2)
    )
    action_prior_per_t = jnp.array([[0.55, 0.45], [0.2, 0.8]], dtype=jnp.float32)
    common = (
        log_transition,
        reduced,
        fwd,
        bwd,
        q_u,
        dyn_cavity,
        theta_prior,
        action_prior,
    )
    return np.asarray(
        [
            _action_cond_entropy(dyn_regions, 2),
            _dyn_cond_entropy(dyn_regions, 2),
            _obs_cond_entropy(obs_regions, 2),
            compute_bethe_vfe_loopy(*common),
            compute_region_extended_vfe(*common, dyn_regions, obs_regions),
            compute_dyn_channel_vfe(*common, dyn_regions),
            compute_vbp_channel_vfe(*common, dyn_regions),
            compute_precise_info_seeking_vfe(*common, dyn_regions, obs_regions),
            compute_active_inference_vfe(*common, dyn_regions, obs_regions),
            compute_nuijten_vfe(
                dyn_regions,
                obs_probabilities,
                fwd,
                bwd,
                q_u,
                tiled_transition,
                obs_factor,
                dyn_cavity,
                obs_cavity,
                theta_prior,
                action_prior_per_t,
            ),
        ],
        dtype=np.float32,
    )


def jax_trace_values() -> dict[str, np.ndarray]:
    state = jnp.array([0.65, 0.35], dtype=jnp.float32)
    static = jnp.array([0.55, 0.45], dtype=jnp.float32)
    transition = jnp.array(
        [
            0.9, 0.2, 0.4, 0.7, 0.3, 0.8, 0.6, 0.1,
            0.1, 0.8, 0.6, 0.3, 0.7, 0.2, 0.4, 0.9,
        ],
        dtype=jnp.float32,
    ).reshape(2, 2, 2, 2)
    observation = jnp.array(
        [0.9, 0.2, 0.3, 0.8, 0.1, 0.8, 0.7, 0.2],
        dtype=jnp.float32,
    ).reshape(1, 2, 2, 2)
    goals = {
        "terminal": jnp.array([0.15, 0.85], dtype=jnp.float32),
        "theta": jnp.array([[0.15, 0.75], [0.85, 0.25]], dtype=jnp.float32),
    }
    action_prior = jnp.array([0.6, 0.4], dtype=jnp.float32)

    def flatten(result: tuple[jnp.ndarray, ...]) -> np.ndarray:
        return np.concatenate(tuple(np.asarray(value).reshape(-1) for value in result))

    traces: dict[str, np.ndarray] = {}
    for goal_name, goal in goals.items():
        traces[f"loopy_{goal_name}"] = flatten(
            loopy_bp_convergence(
                state, static, transition, goal, 2, 2, action_prior=action_prior
            )
        )
        traces[f"region_{goal_name}"] = flatten(
            region_extended_convergence(
                state,
                static,
                transition,
                observation,
                goal,
                2,
                2,
                damping=0.5,
                action_prior=action_prior,
            )
        )
        traces[f"nuijten_{goal_name}"] = flatten(
            nuijten_mp_convergence(
                state,
                static,
                transition,
                observation,
                goal,
                2,
                2,
                action_prior=action_prior,
            )
        )
        traces[f"dyn_{goal_name}"] = flatten(
            dyn_channel_convergence(
                state,
                static,
                transition,
                observation,
                goal,
                2,
                2,
                damping=0.5,
                action_prior=action_prior,
            )
        )
        traces[f"vbp_{goal_name}"] = flatten(
            vbp_channel_convergence(
                state,
                static,
                transition,
                observation,
                goal,
                2,
                2,
                damping=0.5,
                action_prior=action_prior,
            )
        )
        traces[f"precise_{goal_name}"] = flatten(
            precise_info_seeking_convergence(
                state,
                static,
                transition,
                observation,
                goal,
                2,
                2,
                damping=0.5,
                action_prior=action_prior,
            )
        )
        traces[f"active_{goal_name}"] = flatten(
            active_inference_convergence(
                state,
                static,
                transition,
                observation,
                goal,
                2,
                2,
                damping=0.5,
                action_prior=action_prior,
            )
        )
    return traces


def mojo_values() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    completed = subprocess.run(
        [
            "pixi",
            "run",
            "mojo",
            "run",
            "-I",
            "src",
            "tests/mojo/parity_convergence.mojo",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    formula_values = [
        float(line.split()[1])
        for line in completed.stdout.splitlines()
        if line.startswith("convergence_vfe ")
    ]
    indexed_traces: dict[str, dict[int, float]] = {}
    for line in completed.stdout.splitlines():
        if not line.startswith("convergence_trace "):
            continue
        _, name, index, value = line.split()
        indexed_traces.setdefault(name, {})[int(index)] = float(value)
    traces = {
        name: np.asarray(
            [values[index] for index in range(len(values))], dtype=np.float32
        )
        for name, values in indexed_traces.items()
    }
    return np.asarray(formula_values, dtype=np.float32), traces


def main() -> None:
    expected = jax_values()
    expected_traces = jax_trace_values()
    actual, actual_traces = mojo_values()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if actual_traces.keys() != expected_traces.keys():
        raise AssertionError(
            f"trace groups differ: actual={sorted(actual_traces)} "
            f"expected={sorted(expected_traces)}"
        )
    trace_errors: list[float] = []
    trace_values = 0
    for name, expected_trace in expected_traces.items():
        actual_trace = actual_traces[name]
        np.testing.assert_allclose(actual_trace, expected_trace, rtol=1e-5, atol=1e-5)
        trace_values += expected_trace.size
        trace_errors.append(float(np.max(np.abs(actual_trace - expected_trace))))
    errors = np.abs(actual - expected)
    print(
        f"PASS convergence_vfe: values={expected.size} "
        f"max_abs_error={errors.max():.3e}"
    )
    print(
        f"PASS convergence_traces: groups={len(expected_traces)} "
        f"values={trace_values} max_abs_error={max(trace_errors):.3e}"
    )


if __name__ == "__main__":
    main()
