from std.collections import List
from std.testing import TestSuite, assert_true

from aif_mojo.state_inference import (
    state_inference_step,
    state_inference_step_sparse,
)


def assert_close(
    actual: Float32, expected: Float32, tolerance: Float32 = 1.0e-5
) raises:
    assert_true(abs(actual - expected) <= tolerance)


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def test_sparse_state_inference_matches_jax_fixture() raises:
    var transitions = List[Int]()
    transitions.append(0)
    transitions.append(1)
    transitions.append(1)
    transitions.append(1)

    # Shape (fov=1, outcome=2, state=2, static=2).
    var observations = List[Float32]()
    observations.extend(pair(0.9, 0.9))
    observations.extend(pair(0.2, 0.2))
    observations.extend(pair(0.1, 0.1))
    observations.extend(pair(0.8, 0.8))

    # Shape (orientation=2, state=2).
    var orientations = List[Float32]()
    orientations.extend(pair(0.8, 0.3))
    orientations.extend(pair(0.2, 0.7))

    var action = List[Float32]()
    action.append(1.0)
    var result = state_inference_step_sparse(
        pair(0.6, 0.4),
        pair(0.5, 0.5),
        transitions,
        observations,
        orientations,
        pair(1.0, 0.0),
        pair(1.0, 0.0),
        action,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
    )
    # The first S entries are q_current; the next K are q_static.
    assert_close(result[0], 0.91290796)
    assert_close(result[1], 0.08709209)
    assert_close(result[2], 0.8699485)
    assert_close(result[3], 0.13005152)


def test_dense_state_inference_matches_sparse_and_jax() raises:
    # Shape (new=2, old=2, static=2, action=1).
    var transitions = List[Float32]()
    transitions.extend(pair(1.0, 0.0))
    transitions.extend(pair(0.0, 0.0))
    transitions.extend(pair(0.0, 1.0))
    transitions.extend(pair(1.0, 1.0))

    var observations = List[Float32]()
    observations.extend(pair(0.9, 0.9))
    observations.extend(pair(0.2, 0.2))
    observations.extend(pair(0.1, 0.1))
    observations.extend(pair(0.8, 0.8))
    var orientations = List[Float32]()
    orientations.extend(pair(0.8, 0.3))
    orientations.extend(pair(0.2, 0.7))
    var action = List[Float32]()
    action.append(1.0)

    var result = state_inference_step(
        pair(0.6, 0.4),
        pair(0.5, 0.5),
        transitions,
        observations,
        orientations,
        pair(1.0, 0.0),
        pair(1.0, 0.0),
        action,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
    )
    assert_close(result[0], 0.91290796)
    assert_close(result[1], 0.08709209)
    assert_close(result[2], 0.8699485)
    assert_close(result[3], 0.13005152)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
