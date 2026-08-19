from std.collections import List
from std.math import log

from aif_mojo.numerics import EPSILON, softmax


def _copy(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(value)
    return result^


def _zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def state_inference_step_sparse(
    q_old_state: List[Float32],
    q_static_prior: List[Float32],
    transition_indices: List[Int],
    observation_tensors: List[Float32],
    orientation_tensor: List[Float32],
    vision_observation: List[Float32],
    orientation_observation: List[Float32],
    action_onehot: List[Float32],
    n_iterations: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_fov: Int,
    n_outcomes: Int,
    n_orientations: Int,
) -> List[Float32]:
    """Sparse MiniGrid-style state inference; returns q_state followed by q_static.
    """
    debug_assert(
        len(transition_indices) == n_states * n_actions * n_static,
        "transition shape mismatch",
    )
    debug_assert(
        len(observation_tensors) == n_fov * n_outcomes * n_states * n_static,
        "observation tensor shape mismatch",
    )

    var action_idx = Int(0)
    for candidate in range(1, n_actions):
        if action_onehot[candidate] > action_onehot[action_idx]:
            action_idx = candidate

    var q_current = _copy(q_old_state)
    var q_static = _copy(q_static_prior)
    for _ in range(n_iterations):
        var transition_message = _zeros(n_states)
        for old_idx in range(n_states):
            for static_idx in range(n_static):
                var transition_offset = (
                    old_idx * n_actions + action_idx
                ) * n_static + static_idx
                var new_idx = transition_indices[transition_offset]
                transition_message[new_idx] += (
                    q_old_state[old_idx] * q_static[static_idx]
                )
        var transition_total = Float32(0.0)
        for value in transition_message:
            transition_total += value
        for state_idx in range(n_states):
            transition_message[state_idx] /= transition_total + EPSILON

        var log_vision_message = _zeros(n_states)
        for fov_idx in range(n_fov):
            for state_idx in range(n_states):
                var value = Float32(0.0)
                for outcome_idx in range(n_outcomes):
                    for static_idx in range(n_static):
                        var tensor_offset = (
                            (fov_idx * n_outcomes + outcome_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        value += (
                            observation_tensors[tensor_offset]
                            * vision_observation[
                                fov_idx * n_outcomes + outcome_idx
                            ]
                            * q_static[static_idx]
                        )
                log_vision_message[state_idx] += log(value + EPSILON)

        var orientation_message = _zeros(n_states)
        for state_idx in range(n_states):
            for orientation_idx in range(n_orientations):
                orientation_message[state_idx] += (
                    orientation_tensor[orientation_idx * n_states + state_idx]
                    * orientation_observation[orientation_idx]
                )

        var log_current = List[Float32]()
        for state_idx in range(n_states):
            log_current.append(
                log(transition_message[state_idx] + EPSILON)
                + log_vision_message[state_idx]
                + log(orientation_message[state_idx] + EPSILON)
            )
        q_current = softmax(log_current)

        var log_static_message = _zeros(n_static)
        for fov_idx in range(n_fov):
            for static_idx in range(n_static):
                var value = Float32(0.0)
                for outcome_idx in range(n_outcomes):
                    for state_idx in range(n_states):
                        var tensor_offset = (
                            (fov_idx * n_outcomes + outcome_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        value += (
                            observation_tensors[tensor_offset]
                            * vision_observation[
                                fov_idx * n_outcomes + outcome_idx
                            ]
                            * q_current[state_idx]
                        )
                log_static_message[static_idx] += log(value + EPSILON)

        for static_idx in range(n_static):
            var transition_value = Float32(0.0)
            for old_idx in range(n_states):
                var transition_offset = (
                    old_idx * n_actions + action_idx
                ) * n_static + static_idx
                var new_idx = transition_indices[transition_offset]
                transition_value += q_current[new_idx] * q_old_state[old_idx]
            log_static_message[static_idx] += log(
                transition_value + EPSILON
            ) + log(q_static_prior[static_idx] + EPSILON)
        q_static = softmax(log_static_message)

    var result = List[Float32]()
    for value in q_current:
        result.append(value)
    for value in q_static:
        result.append(value)
    return result^


def state_inference_step(
    q_old_state: List[Float32],
    q_static_prior: List[Float32],
    transition_tensor: List[Float32],
    observation_tensors: List[Float32],
    orientation_tensor: List[Float32],
    vision_observation: List[Float32],
    orientation_observation: List[Float32],
    action_onehot: List[Float32],
    n_iterations: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_fov: Int,
    n_outcomes: Int,
    n_orientations: Int,
) -> List[Float32]:
    """Dense state inference; returns q_state followed by q_static."""
    debug_assert(
        len(transition_tensor) == n_states * n_states * n_static * n_actions,
        "transition tensor shape mismatch",
    )
    var q_current = _copy(q_old_state)
    var q_static = _copy(q_static_prior)
    for _ in range(n_iterations):
        var transition_message = _zeros(n_states)
        for new_idx in range(n_states):
            for old_idx in range(n_states):
                for static_idx in range(n_static):
                    for action_idx in range(n_actions):
                        var offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        transition_message[new_idx] += (
                            transition_tensor[offset]
                            * q_old_state[old_idx]
                            * q_static[static_idx]
                            * action_onehot[action_idx]
                        )
        var transition_total = Float32(0.0)
        for value in transition_message:
            transition_total += value
        for state_idx in range(n_states):
            transition_message[state_idx] /= transition_total + EPSILON

        var log_vision_message = _zeros(n_states)
        for fov_idx in range(n_fov):
            for state_idx in range(n_states):
                var value = Float32(0.0)
                for outcome_idx in range(n_outcomes):
                    for static_idx in range(n_static):
                        var tensor_offset = (
                            (fov_idx * n_outcomes + outcome_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        value += (
                            observation_tensors[tensor_offset]
                            * vision_observation[
                                fov_idx * n_outcomes + outcome_idx
                            ]
                            * q_static[static_idx]
                        )
                log_vision_message[state_idx] += log(value + EPSILON)

        var orientation_message = _zeros(n_states)
        for state_idx in range(n_states):
            for orientation_idx in range(n_orientations):
                orientation_message[state_idx] += (
                    orientation_tensor[orientation_idx * n_states + state_idx]
                    * orientation_observation[orientation_idx]
                )

        var log_current = List[Float32]()
        for state_idx in range(n_states):
            log_current.append(
                log(transition_message[state_idx] + EPSILON)
                + log_vision_message[state_idx]
                + log(orientation_message[state_idx] + EPSILON)
            )
        q_current = softmax(log_current)

        var log_static_message = _zeros(n_static)
        for fov_idx in range(n_fov):
            for static_idx in range(n_static):
                var value = Float32(0.0)
                for outcome_idx in range(n_outcomes):
                    for state_idx in range(n_states):
                        var tensor_offset = (
                            (fov_idx * n_outcomes + outcome_idx) * n_states
                            + state_idx
                        ) * n_static + static_idx
                        value += (
                            observation_tensors[tensor_offset]
                            * vision_observation[
                                fov_idx * n_outcomes + outcome_idx
                            ]
                            * q_current[state_idx]
                        )
                log_static_message[static_idx] += log(value + EPSILON)

        for static_idx in range(n_static):
            var transition_value = Float32(0.0)
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    for action_idx in range(n_actions):
                        var offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        transition_value += (
                            transition_tensor[offset]
                            * q_current[new_idx]
                            * q_old_state[old_idx]
                            * action_onehot[action_idx]
                        )
            log_static_message[static_idx] += log(
                transition_value + EPSILON
            ) + log(q_static_prior[static_idx] + EPSILON)
        q_static = softmax(log_static_message)

    var result = List[Float32]()
    for value in q_current:
        result.append(value)
    for value in q_static:
        result.append(value)
    return result^
