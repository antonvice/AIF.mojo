"""Bounded warm-run benchmark against the current UAI-MP-AIF-JAX APIs."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from environments.minigrid import (
    generate_observation_tensor,
    generate_transition_tensor,
    get_valid_static_configs,
)
from inference.active_inference import active_inference_planning
from inference.dyn_channel_loopy_bp import dyn_channel_loopy_bp_planning
from inference.loopy_bp import loopy_bp_planning
from inference.loopy_vbp import loopy_vbp_planning
from inference.nuijten_mp import nuijten_mp_planning
from inference.precise_info_seeking import precise_info_seeking_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.vbp_channel import vbp_channel_planning


def block_ready(value):
    if isinstance(value, jax.Array):
        value.block_until_ready()
    elif isinstance(value, tuple):
        for item in value:
            block_ready(item)


def benchmark(name, fn, args, runs):
    start = time.perf_counter()
    block_ready(fn(*args))
    compile_ms = (time.perf_counter() - start) * 1000

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        block_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1000)

    q25, median, q75 = np.percentile(samples, [25, 50, 75])
    print(f"{name}\t{compile_ms:.3f}\t{median:.3f}\t{q75 - q25:.3f}")


def main():
    grid_size = 4
    fov_size = 3
    horizon = 5
    iterations = 3
    runs = 10

    valid_configs = get_valid_static_configs(grid_size)
    transition = jnp.array(
        generate_transition_tensor(grid_size, valid_configs), dtype=jnp.float32
    )
    observation = jnp.array(
        generate_observation_tensor(grid_size, valid_configs, fov_size=fov_size),
        dtype=jnp.float32,
    )
    observation = observation.reshape(
        fov_size * fov_size, *observation.shape[2:]
    )

    states = transition.shape[0]
    static = transition.shape[2]
    q_state = jnp.ones(states) / states
    q_static = jnp.ones(static) / static
    goal = jnp.zeros(states).at[0].set(1.0)

    common = (q_state, q_static, transition)
    planners = (
        ("loopy-vbp", loopy_vbp_planning, (*common, goal, horizon, iterations)),
        ("loopy", loopy_bp_planning, (*common, goal, horizon, iterations)),
        (
            "region-extended",
            region_extended_loopy_bp_planning,
            (*common, observation, goal, horizon, iterations),
        ),
        (
            "dyn-channel",
            dyn_channel_loopy_bp_planning,
            (*common, observation, goal, horizon, iterations),
        ),
        (
            "nuijten",
            nuijten_mp_planning,
            (*common, observation, goal, horizon, iterations),
        ),
        (
            "vbp-channel",
            vbp_channel_planning,
            (*common, observation, goal, horizon, iterations),
        ),
        (
            "precise-info-seeking",
            precise_info_seeking_planning,
            (*common, observation, goal, horizon, iterations),
        ),
        (
            "active-inference",
            active_inference_planning,
            (*common, observation, goal, horizon, iterations),
        ),
    )

    print(f"backend\t{jax.default_backend()}")
    print(
        f"config\tgrid={grid_size}\tstates={states}\tstatic={static}"
        f"\tactions={transition.shape[3]}\thorizon={horizon}"
        f"\titerations={iterations}\truns={runs}"
    )
    print("planner\tcompile_ms\tmedian_ms\tiqr_ms")
    for name, fn, args in planners:
        benchmark(name, fn, args, runs)


if __name__ == "__main__":
    main()
