from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_true

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
from aif_mojo.numerics import safe_log


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def make_list(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def test_forward_message_2d_contracts_and_normalizes() raises:
    var tensor = List[Float32]()
    tensor.append(0.8)
    tensor.append(0.1)
    tensor.append(0.2)
    tensor.append(0.9)
    var result = forward_message_2d(tensor, 2, 2, make_list(0.25, 0.75))
    assert_close(result[0], 0.275)
    assert_close(result[1], 0.725)


def test_backward_message_2d_is_unnormalized() raises:
    var tensor = List[Float32]()
    tensor.append(0.9)
    tensor.append(0.2)
    tensor.append(0.1)
    tensor.append(0.8)
    var result = backward_message_2d(tensor, 2, 2, make_list(1.0, 0.0))
    assert_close(result[0], 0.9)
    assert_close(result[1], 0.2)


def test_three_dimensional_xor_messages() raises:
    var tensor = List[Float32]()
    # out=0 (equal), then out=1 (different), row-major.
    tensor.append(1.0)
    tensor.append(0.0)
    tensor.append(0.0)
    tensor.append(1.0)
    tensor.append(0.0)
    tensor.append(1.0)
    tensor.append(1.0)
    tensor.append(0.0)

    var q_state = make_list(0.25, 0.75)
    var q_other = make_list(0.6, 0.4)
    var forward = forward_message_3d(tensor, 2, 2, 2, q_state, q_other)
    assert_close(forward[0], 0.45)
    assert_close(forward[1], 0.55)

    var backward = backward_message_3d(
        tensor, 2, 2, 2, make_list(1.0, 0.0), q_other
    )
    assert_close(backward[0], 0.6)
    assert_close(backward[1], 0.4)

    var to_other = backward_message_to_other_3d(
        tensor, 2, 2, 2, make_list(1.0, 0.0), q_state
    )
    assert_close(to_other[0], 0.25)
    assert_close(to_other[1], 0.75)


def test_forward_message_4d_contracts_all_inputs() raises:
    var tensor = List[Float32]()
    # out=0 for even parity, out=1 for odd parity.
    tensor.extend(make_list(1.0, 0.0))
    tensor.extend(make_list(0.0, 1.0))
    tensor.extend(make_list(0.0, 1.0))
    tensor.extend(make_list(1.0, 0.0))
    tensor.extend(make_list(0.0, 1.0))
    tensor.extend(make_list(1.0, 0.0))
    tensor.extend(make_list(1.0, 0.0))
    tensor.extend(make_list(0.0, 1.0))
    var result = forward_message_4d(
        tensor,
        2,
        2,
        2,
        2,
        make_list(0.2, 0.8),
        make_list(0.3, 0.7),
        make_list(0.4, 0.6),
    )
    assert_close(result[0], 0.476)
    assert_close(result[1], 0.524)


def test_marginalize_static_matches_weighted_transition() raises:
    var log_tensor = List[Float32]()
    log_tensor.append(safe_log(0.9))
    log_tensor.append(safe_log(0.2))
    log_tensor.append(safe_log(0.1))
    log_tensor.append(safe_log(0.8))
    var result = marginalize_static(
        log_tensor, 2, 1, 2, 1, make_list(safe_log(0.25), safe_log(0.75))
    )
    assert_close(exp(result[0]), 0.375)
    assert_close(exp(result[1]), 0.625)


def test_combine_messages_matches_normalized_product() raises:
    var messages = List[Float32]()
    messages.extend(make_list(0.8, 0.2))
    messages.extend(make_list(0.6, 0.4))
    var result = combine_messages(messages, 2, 2)
    assert_close(result[0], 0.85714287)
    assert_close(result[1], 0.14285715)


def test_combine_log_messages_matches_probability_variant() raises:
    var messages = List[Float32]()
    messages.append(safe_log(0.8))
    messages.append(safe_log(0.2))
    messages.append(safe_log(0.6))
    messages.append(safe_log(0.4))
    var result = combine_messages_log(messages, 2, 2)
    assert_close(result[0], 0.85714287)
    assert_close(result[1], 0.14285715)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
