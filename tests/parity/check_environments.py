"""Compare complete native environment tensors against the frozen JAX oracle."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import nn
from jax.scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "UAI-MP-AIF-JAX"))

from environments.frozen_lake import (  # noqa: E402
    generate_goal,
    generate_observation_tensor,
    generate_transition_tensor,
)
from agents.frozen_lake_agent import _infer_state  # noqa: E402
from agents.flat_tensor_agent import ActiveInferenceAgent  # noqa: E402
from agents.rocksample_agent import _infer_state as infer_rocksample  # noqa: E402
from agents.wumpus_agent import _infer_state as infer_wumpus  # noqa: E402
from inference.loopy_vbp import loopy_vbp_planning  # noqa: E402
from inference.loopy_bp import loopy_bp_planning  # noqa: E402
from inference.messages import safe_log  # noqa: E402
from environments.wumpus_world import (  # noqa: E402
    generate_goal as generate_wumpus_goal,
    generate_observation_tensor as generate_wumpus_observation,
    generate_transition_tensor as generate_wumpus_transition,
)
from environments.rocksample import (  # noqa: E402
    all_quality_configs,
    generate_goal as generate_rocksample_goal,
    generate_observation_tensor as generate_rocksample_observation,
    generate_transition_tensor as generate_rocksample_transition,
)
from environments.minigrid import (  # noqa: E402
    ActionType,
    N_DOOR_KEY_STATES,
    N_ORIENTATIONS,
    Orientation,
    door_position,
    direction_to_onehot,
    flatten_state_index,
    generate_observation_tensor as generate_minigrid_observation,
    generate_orientation_observation_tensor,
    generate_transition_tensor as generate_minigrid_transition,
    get_fov,
    get_next_agent_position,
    get_next_door_key_state,
    get_next_orientation,
    key_position,
    observation_to_onehot,
    soften_observation_tensor,
    state_to_coords,
)


def expected_outputs() -> dict[str, np.ndarray]:
    holes = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    dense = generate_transition_tensor(2, holes, slip_prob=0.0, goal_pos=3)
    indices = np.argmax(dense, axis=0).transpose(0, 2, 1)
    observation = generate_observation_tensor(
        2, holes, base_noise=0.05, noise_range=0.15
    )
    frozen_observation_sample = (
        np.full(12, 0.5, dtype=np.float32) < observation[:, 1, 4, 0]
    ).astype(np.float32)
    frozen_simulator_step = np.concatenate(
        (
            np.array([4, 1, 0, 0, 0], dtype=np.float32),
            frozen_observation_sample,
        )
    )
    goal = generate_goal(
        2,
        holes,
        goal_pos=3,
        goal_reward=1.0,
        hole_penalty=1.0,
        temperature=1.0,
    )
    start = np.zeros(8, dtype=np.float32)
    start[0] = 1.0
    static = np.array([0.5, 0.5], dtype=np.float32)
    observation_before_scan = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32
    )
    observation_after_scan = np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32
    )
    first_state, first_static = _infer_state(
        start,
        static,
        dense,
        observation,
        observation_before_scan,
        np.ones(5, dtype=np.float32) / 5,
    )
    second_state, second_static = _infer_state(
        first_state,
        first_static,
        dense,
        observation,
        observation_after_scan,
        np.eye(5, dtype=np.float32)[4],
    )
    frozen_action = loopy_bp_planning(
        first_state,
        first_static,
        dense,
        goal,
        horizon=2,
        n_iterations=2,
        action_prior=np.ones(5, dtype=np.float32) / 5,
    )
    wumpus_pits = np.array(
        [[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32
    )
    wumpus_locations = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    wumpus_gold = np.array(
        [[0, 0, 0, 1], [0, 1, 0, 0]], dtype=np.float32
    )
    wumpus_observation = generate_wumpus_observation(
        2,
        wumpus_pits,
        wumpus_locations,
        wumpus_gold,
        obs_noise=0.1,
        pos_noise=0.1,
    )
    wumpus_draws = np.array(
        [0.5, 0.005, 0.02, 0.95, 0.001, 0.5, 0.2], dtype=np.float32
    )
    wumpus_sample = (
        wumpus_draws < wumpus_observation[:, 1, 4, 0]
    ).astype(np.float32)
    wumpus_state_prior = np.zeros(8, dtype=np.float32)
    wumpus_state_prior[0] = 1.0
    wumpus_static_prior = np.full(2, 0.5, dtype=np.float32)
    wumpus_agent_observation = np.array(
        [0, 0, 0, 1, 0, 0, 0], dtype=np.float32
    )
    wumpus_q_state, wumpus_q_static = infer_wumpus(
        wumpus_state_prior,
        wumpus_static_prior,
        generate_wumpus_transition(
            2, wumpus_pits, wumpus_locations, slip_prob=0.0
        ),
        wumpus_observation,
        np.eye(2, dtype=np.float32)[wumpus_agent_observation.astype(np.int32)],
        np.full(5, 0.2, dtype=np.float32),
    )
    wumpus_agent_goal = generate_wumpus_goal(
        2,
        wumpus_pits,
        wumpus_locations,
        wumpus_gold,
        gold_reward=1.0,
        pit_penalty=1.0,
        wumpus_penalty=1.0,
        temperature=1.0,
    )
    wumpus_action = loopy_bp_planning(
        wumpus_q_state,
        wumpus_q_static,
        generate_wumpus_transition(
            2, wumpus_pits, wumpus_locations, slip_prob=0.0
        ),
        wumpus_agent_goal,
        horizon=2,
        n_iterations=1,
        action_prior=np.full(5, 0.2, dtype=np.float32),
    )
    wumpus_agent_step = np.concatenate(
        (
            np.array([np.argmax(wumpus_action), 2], dtype=np.float32),
            np.asarray(wumpus_q_state),
            np.asarray(wumpus_q_static),
        )
    )
    rock_positions = np.array([0], dtype=np.int32)
    rock_qualities = all_quality_configs(1)
    rock_transition = generate_rocksample_transition(
        2, rock_positions, 1, slip_prob=0.2
    )
    rock_deterministic = generate_rocksample_transition(
        2, rock_positions, 1, slip_prob=0.0
    )
    rock_indices = np.argmax(rock_deterministic, axis=0).transpose(0, 2, 1)
    rock_observation = generate_rocksample_observation(
        2,
        rock_positions,
        rock_qualities,
        1,
        half_eff_dist=2.0,
        pos_noise=0.1,
    )
    rock_probabilities = np.asarray(rock_observation[:, :, 10, 1])
    rock_thresholds = np.full(5, 0.5, dtype=np.float32) * np.sum(
        rock_probabilities, axis=1
    )
    rock_observation_sample = np.asarray(
        [
            np.searchsorted(
                np.cumsum(rock_probabilities[channel]),
                rock_thresholds[channel],
                side="right",
            )
            for channel in range(5)
        ],
        dtype=np.float32,
    )
    rock_simulator_step = np.concatenate(
        (
            np.array([10, 1, 0, 0, 0], dtype=np.float32),
            rock_observation_sample,
        )
    )
    rock_goal = generate_rocksample_goal(
        2,
        rock_positions,
        rock_qualities,
        1,
        good_logit=2.0,
        bad_logit=4.0,
        exit_logit=2.0,
        temperature=1.0,
    )
    rock_state_prior = np.zeros(24, dtype=np.float32)
    rock_state_prior[2] = 1.0
    rock_static_prior = np.full(2, 0.5, dtype=np.float32)
    rock_agent_observation = np.array([0, 0, 1, 0, 2], dtype=np.float32)
    rock_q_state_1, rock_q_static_1 = infer_rocksample(
        rock_state_prior,
        rock_static_prior,
        rock_deterministic,
        rock_observation,
        rock_agent_observation,
        np.full(6, 1.0 / 6.0, dtype=np.float32),
    )
    rock_terminal_goal_1 = nn.softmax(
        logsumexp(
            safe_log(rock_goal) + safe_log(rock_q_static_1)[None, :], axis=1
        )
    )
    rock_action_1 = loopy_vbp_planning(
        rock_q_state_1,
        rock_q_static_1,
        rock_deterministic,
        rock_terminal_goal_1,
        horizon=2,
        n_iterations=1,
    )
    rock_agent_step_1 = np.concatenate(
        (
            np.array([np.argmax(rock_action_1), 2], dtype=np.float32),
            np.asarray(rock_q_state_1),
            np.asarray(rock_q_static_1),
        )
    )
    rock_q_state_2, rock_q_static_2 = infer_rocksample(
        rock_q_state_1,
        rock_q_static_1,
        rock_deterministic,
        rock_observation,
        rock_agent_observation,
        np.eye(6, dtype=np.float32)[0],
    )
    rock_terminal_goal_2 = nn.softmax(
        logsumexp(
            safe_log(rock_goal) + safe_log(rock_q_static_2)[None, :], axis=1
        )
    )
    rock_action_2 = loopy_vbp_planning(
        rock_q_state_2,
        rock_q_static_2,
        rock_deterministic,
        rock_terminal_goal_2,
        horizon=1,
        n_iterations=1,
    )
    rock_agent_step_2 = np.concatenate(
        (
            np.array([np.argmax(rock_action_2), 1], dtype=np.float32),
            np.asarray(rock_q_state_2),
            np.asarray(rock_q_static_2),
        )
    )
    minigrid_configs = [(1, 1)]
    minigrid_transition = generate_minigrid_transition(
        3, minigrid_configs, dtype=np.float32
    )
    minigrid_indices = np.argmax(minigrid_transition, axis=0).transpose(0, 2, 1)
    minigrid_hard = generate_minigrid_observation(
        3, minigrid_configs, fov_size=3, dtype=np.float32
    )
    minigrid_soft = soften_observation_tensor(minigrid_hard, 3, 0.1)
    minigrid_goal = np.zeros(108, dtype=np.float32)
    for orientation in range(N_ORIENTATIONS):
        goal_state = flatten_state_index(
            8, orientation, 2, 9, N_ORIENTATIONS, N_DOOR_KEY_STATES
        )
        minigrid_goal[goal_state] = 0.25
    minigrid_agent = ActiveInferenceAgent.create(
        grid_size=3,
        transition_tensor=None,
        observation_tensors=minigrid_hard,
        orientation_tensor=generate_orientation_observation_tensor(
            3, dtype=np.float32
        ),
        goal=minigrid_goal,
        planning_horizon=3,
        n_inference_iterations=1,
        n_planning_iterations=1,
        damping=1.0,
        T_idx=jnp.asarray(minigrid_indices, dtype=jnp.int32),
    )
    minigrid_action, minigrid_updated = minigrid_agent.step(
        observation_to_onehot(
            get_fov(
                0,
                0,
                Orientation.DOWN,
                0,
                1,
                1,
                1,
                0,
                3,
                fov_size=3,
            )
        ),
        direction_to_onehot(Orientation.DOWN),
        time_remaining=1,
    )
    minigrid_agent_step = np.concatenate(
        (
            np.array([minigrid_action, 1], dtype=np.float32),
            np.asarray(minigrid_updated.q_state),
            np.asarray(minigrid_updated.q_static),
        )
    )
    minigrid_state = flatten_state_index(
        0, Orientation.DOWN, 0, 9, N_ORIENTATIONS, N_DOOR_KEY_STATES
    )
    minigrid_tape = [minigrid_state]
    for minigrid_action in [
        ActionType.PICKUP,
        ActionType.FORWARD,
        ActionType.TURN_LEFT,
        ActionType.TOGGLE,
        ActionType.FORWARD,
        ActionType.FORWARD,
        ActionType.TURN_RIGHT,
        ActionType.FORWARD,
    ]:
        location = minigrid_state // (N_ORIENTATIONS * N_DOOR_KEY_STATES)
        orientation = (minigrid_state // N_DOOR_KEY_STATES) % N_ORIENTATIONS
        door_key_state = minigrid_state % N_DOOR_KEY_STATES
        agent_x, agent_y = state_to_coords(location, 3)
        key_x, key_y = key_position(1, 3)
        door_x, door_y = door_position(1, 3)
        next_location = get_next_agent_position(
            agent_x,
            agent_y,
            orientation,
            door_x,
            door_y,
            key_x,
            key_y,
            door_key_state,
            minigrid_action,
            3,
        )
        next_door_key = get_next_door_key_state(
            agent_x,
            agent_y,
            orientation,
            key_x,
            key_y,
            door_x,
            door_y,
            minigrid_action,
            door_key_state,
        )
        minigrid_state = flatten_state_index(
            next_location,
            get_next_orientation(orientation, minigrid_action),
            next_door_key,
            9,
            N_ORIENTATIONS,
            N_DOOR_KEY_STATES,
        )
        minigrid_tape.append(minigrid_state)
    return {
        "frozen_transition": dense.reshape(-1),
        "frozen_transition_slip": generate_transition_tensor(
            2, holes, slip_prob=0.3, goal_pos=3
        ).reshape(-1),
        "frozen_transition_indices": indices.reshape(-1),
        "frozen_observation": observation.reshape(-1),
        "frozen_observation_sample": frozen_observation_sample,
        "frozen_simulator_step": frozen_simulator_step,
        "frozen_goal": goal.reshape(-1),
        "frozen_tape": np.array([0, 4, 6, 7], dtype=np.float32),
        "frozen_agent_belief_1": np.concatenate((first_state, first_static)),
        "frozen_agent_belief_2": np.concatenate((second_state, second_static)),
        "frozen_agent_step": np.concatenate(
            (
                np.array([np.argmax(frozen_action), 2], dtype=np.float32),
                first_state,
                first_static,
            )
        ),
        "wumpus_transition": generate_wumpus_transition(
            2, wumpus_pits, wumpus_locations, slip_prob=0.0
        ).reshape(-1),
        "wumpus_transition_slip": generate_wumpus_transition(
            2, wumpus_pits, wumpus_locations, slip_prob=0.3
        ).reshape(-1),
        "wumpus_observation": wumpus_observation.reshape(-1),
        "wumpus_goal": generate_wumpus_goal(
            2,
            wumpus_pits,
            wumpus_locations,
            wumpus_gold,
            gold_reward=1.0,
            pit_penalty=1.0,
            wumpus_penalty=1.0,
            temperature=1.0,
        ).reshape(-1),
        "wumpus_sample": wumpus_sample,
        "wumpus_tape": np.array([0, 4, 1, 1, 1], dtype=np.float32),
        "wumpus_agent_step": wumpus_agent_step,
        "rocksample_qualities": rock_qualities.reshape(-1),
        "rocksample_transition": rock_transition.reshape(-1),
        "rocksample_transition_indices": rock_indices.reshape(-1),
        "rocksample_observation": rock_observation.reshape(-1),
        "rocksample_observation_sample": rock_observation_sample,
        "rocksample_simulator_step": rock_simulator_step,
        "rocksample_goal": rock_goal.reshape(-1),
        "rocksample_tape": np.array([2, 0, 8, 20, 6, 7], dtype=np.float32),
        "rocksample_agent_step_1": rock_agent_step_1,
        "rocksample_agent_step_2": rock_agent_step_2,
        "minigrid_transition_indices": minigrid_indices.reshape(-1),
        "minigrid_transition": minigrid_transition.reshape(-1),
        "minigrid_observation": minigrid_hard.reshape(-1),
        "minigrid_observation_soft": minigrid_soft.reshape(-1),
        "minigrid_orientation": generate_orientation_observation_tensor(
            3, dtype=np.float32
        ).reshape(-1),
        "minigrid_goal": minigrid_goal,
        "minigrid_agent_step": minigrid_agent_step,
        "minigrid_tape": np.asarray(minigrid_tape, dtype=np.float32),
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
            "tests/mojo/parity_environments.mojo",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[str, list[float]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            values.setdefault(parts[0], []).append(float(parts[1]))
    return {name: np.asarray(items, dtype=np.float32) for name, items in values.items()}


def main() -> None:
    expected = expected_outputs()
    actual = mojo_outputs()
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"Mojo output keys differ: expected {sorted(expected)}, got {sorted(actual)}"
        )
    for name, expected_value in expected.items():
        np.testing.assert_allclose(actual[name], expected_value, rtol=1e-5, atol=1e-6)
        error = np.max(np.abs(actual[name] - expected_value))
        print(f"PASS {name}: values={expected_value.size} max_abs_error={error:.3e}")


if __name__ == "__main__":
    main()
