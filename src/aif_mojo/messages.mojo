from std.collections import List

from aif_mojo.numerics import EPSILON, logsumexp, safe_log, softmax


def forward_message_2d(
    tensor: List[Float32], n_out: Int, n_in: Int, q_in: List[Float32]
) -> List[Float32]:
    """Contract a row-major (n_out, n_in) tensor and normalize the result."""
    debug_assert(len(tensor) == n_out * n_in, "tensor shape mismatch")
    debug_assert(len(q_in) == n_in, "input shape mismatch")

    var result = List[Float32]()
    var total = Float32(0.0)
    for out_idx in range(n_out):
        var value = Float32(0.0)
        for in_idx in range(n_in):
            value += tensor[out_idx * n_in + in_idx] * q_in[in_idx]
        result.append(value)
        total += value

    var denominator = total + EPSILON
    for i in range(n_out):
        result[i] /= denominator
    return result^


def forward_message_3d(
    tensor: List[Float32],
    n_out: Int,
    n_in1: Int,
    n_in2: Int,
    q_in1: List[Float32],
    q_in2: List[Float32],
) -> List[Float32]:
    """Contract a row-major (n_out, n_in1, n_in2) tensor and normalize."""
    debug_assert(len(tensor) == n_out * n_in1 * n_in2, "tensor shape mismatch")
    debug_assert(len(q_in1) == n_in1, "first input shape mismatch")
    debug_assert(len(q_in2) == n_in2, "second input shape mismatch")

    var result = List[Float32]()
    var total = Float32(0.0)
    for out_idx in range(n_out):
        var value = Float32(0.0)
        for in1_idx in range(n_in1):
            for in2_idx in range(n_in2):
                var offset = (out_idx * n_in1 + in1_idx) * n_in2 + in2_idx
                value += tensor[offset] * q_in1[in1_idx] * q_in2[in2_idx]
        result.append(value)
        total += value

    var denominator = total + EPSILON
    for i in range(n_out):
        result[i] /= denominator
    return result^


def forward_message_4d(
    tensor: List[Float32],
    n_out: Int,
    n_in1: Int,
    n_in2: Int,
    n_in3: Int,
    q_in1: List[Float32],
    q_in2: List[Float32],
    q_in3: List[Float32],
) -> List[Float32]:
    """Contract a row-major 4-D tensor and normalize the output message."""
    debug_assert(
        len(tensor) == n_out * n_in1 * n_in2 * n_in3,
        "tensor shape mismatch",
    )
    debug_assert(len(q_in1) == n_in1, "first input shape mismatch")
    debug_assert(len(q_in2) == n_in2, "second input shape mismatch")
    debug_assert(len(q_in3) == n_in3, "third input shape mismatch")

    var result = List[Float32]()
    var total = Float32(0.0)
    for out_idx in range(n_out):
        var value = Float32(0.0)
        for in1_idx in range(n_in1):
            for in2_idx in range(n_in2):
                for in3_idx in range(n_in3):
                    var offset = (
                        (out_idx * n_in1 + in1_idx) * n_in2 + in2_idx
                    ) * n_in3 + in3_idx
                    value += (
                        tensor[offset]
                        * q_in1[in1_idx]
                        * q_in2[in2_idx]
                        * q_in3[in3_idx]
                    )
        result.append(value)
        total += value

    var denominator = total + EPSILON
    for i in range(n_out):
        result[i] /= denominator
    return result^


def backward_message_2d(
    tensor: List[Float32], n_obs: Int, n_state: Int, obs_onehot: List[Float32]
) -> List[Float32]:
    """Contract an observation from row-major (n_obs, n_state), unnormalized."""
    debug_assert(len(tensor) == n_obs * n_state, "tensor shape mismatch")
    debug_assert(len(obs_onehot) == n_obs, "observation shape mismatch")

    var result = List[Float32]()
    for state_idx in range(n_state):
        var value = Float32(0.0)
        for obs_idx in range(n_obs):
            value += tensor[obs_idx * n_state + state_idx] * obs_onehot[obs_idx]
        result.append(value)
    return result^


def backward_message_3d(
    tensor: List[Float32],
    n_obs: Int,
    n_state: Int,
    n_other: Int,
    obs_onehot: List[Float32],
    q_other: List[Float32],
) -> List[Float32]:
    """Contract observation and other-state beliefs, leaving state unnormalized.
    """
    debug_assert(
        len(tensor) == n_obs * n_state * n_other, "tensor shape mismatch"
    )
    debug_assert(len(obs_onehot) == n_obs, "observation shape mismatch")
    debug_assert(len(q_other) == n_other, "other-state shape mismatch")

    var result = List[Float32]()
    for state_idx in range(n_state):
        var value = Float32(0.0)
        for obs_idx in range(n_obs):
            for other_idx in range(n_other):
                var offset = (
                    obs_idx * n_state + state_idx
                ) * n_other + other_idx
                value += (
                    tensor[offset] * obs_onehot[obs_idx] * q_other[other_idx]
                )
        result.append(value)
    return result^


def backward_message_to_other_3d(
    tensor: List[Float32],
    n_obs: Int,
    n_state: Int,
    n_other: Int,
    obs_onehot: List[Float32],
    q_state: List[Float32],
) -> List[Float32]:
    """Contract observation and state beliefs, leaving other-state unnormalized.
    """
    debug_assert(
        len(tensor) == n_obs * n_state * n_other, "tensor shape mismatch"
    )
    debug_assert(len(obs_onehot) == n_obs, "observation shape mismatch")
    debug_assert(len(q_state) == n_state, "state shape mismatch")

    var result = List[Float32]()
    for other_idx in range(n_other):
        var value = Float32(0.0)
        for obs_idx in range(n_obs):
            for state_idx in range(n_state):
                var offset = (
                    obs_idx * n_state + state_idx
                ) * n_other + other_idx
                value += (
                    tensor[offset] * obs_onehot[obs_idx] * q_state[state_idx]
                )
        result.append(value)
    return result^


def combine_messages(
    messages: List[Float32], n_messages: Int, width: Int
) -> List[Float32]:
    """Combine row-major probability messages via normalized product."""
    debug_assert(len(messages) == n_messages * width, "message shape mismatch")
    var combined_logs = List[Float32]()
    for value_idx in range(width):
        var value = Float32(0.0)
        for message_idx in range(n_messages):
            value += safe_log(
                messages[message_idx * width + value_idx] + EPSILON
            )
        combined_logs.append(value)
    return softmax(combined_logs)


def marginalize_static(
    log_tensor: List[Float32],
    n_new: Int,
    n_old: Int,
    n_static: Int,
    n_actions: Int,
    log_q_static: List[Float32],
) -> List[Float32]:
    """Log-sum-exp over the static axis of (new, old, static, action)."""
    debug_assert(
        len(log_tensor) == n_new * n_old * n_static * n_actions,
        "tensor shape mismatch",
    )
    debug_assert(len(log_q_static) == n_static, "static belief shape mismatch")

    var result = List[Float32]()
    for new_idx in range(n_new):
        for old_idx in range(n_old):
            for action_idx in range(n_actions):
                var terms = List[Float32]()
                for static_idx in range(n_static):
                    var offset = (
                        (new_idx * n_old + old_idx) * n_static + static_idx
                    ) * n_actions + action_idx
                    terms.append(log_tensor[offset] + log_q_static[static_idx])
                result.append(logsumexp(terms))
    return result^


def combine_messages_log(
    log_messages: List[Float32], n_messages: Int, width: Int
) -> List[Float32]:
    """Combine row-major log messages via addition and softmax."""
    debug_assert(
        len(log_messages) == n_messages * width, "message shape mismatch"
    )
    var combined_logs = List[Float32]()
    for value_idx in range(width):
        var value = Float32(0.0)
        for message_idx in range(n_messages):
            value += log_messages[message_idx * width + value_idx]
        combined_logs.append(value)
    return softmax(combined_logs)
