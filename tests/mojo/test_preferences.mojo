from std.collections import List
from std.math import exp
from std.testing import TestSuite, assert_true

from aif_mojo.preferences import (
    combine_preference_messages,
    observation_preference_messages,
    preference_time_slice,
    state_preference_messages,
)


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def test_shared_terminal_and_time_indexed_state_preferences() raises:
    var terminal = state_preference_messages(pair(0.25, 0.75), 2, 2, True)
    assert_true(len(terminal) == 6)
    assert_true(terminal[0] == 0.0 and terminal[3] == 0.0)
    assert_true(abs(exp(terminal[4]) - 0.25) < 1.0e-6)
    var scheduled = List[Float32]()
    scheduled.extend(pair(0.5, 0.5))
    scheduled.extend(pair(0.8, 0.2))
    scheduled.extend(pair(0.1, 0.9))
    var messages = state_preference_messages(scheduled, 2, 2)
    var middle = preference_time_slice(messages, 1, 2)
    assert_true(abs(exp(middle[0]) - 0.8) < 1.0e-6)
    assert_true(abs(exp(middle[1]) - 0.2) < 1.0e-6)


def test_observation_preferences_map_to_state_messages() raises:
    # One field, two outcomes, two states, one theta: state 0 emits outcome 0.
    var observation = List[Float32]()
    observation.extend(pair(1.0, 0.0))
    observation.extend(pair(0.0, 1.0))
    var q_static = List[Float32]()
    q_static.append(1.0)
    var result = observation_preference_messages(
        observation, q_static, pair(0.9, 0.1), 1, 1, 2, 2, 1
    )
    assert_true(len(result) == 4)
    assert_true(abs(exp(result[0]) - 0.9) < 1.0e-6)
    assert_true(abs(exp(result[1]) - 0.1) < 1.0e-6)
    assert_true(abs(result[0] - result[2]) < 1.0e-6)


def test_combined_preference_messages_add_log_factors() raises:
    var combined = combine_preference_messages(
        pair(-0.2, -0.4), pair(-0.3, -0.6)
    )
    assert_true(abs(combined[0] + 0.5) < 1.0e-6)
    assert_true(abs(combined[1] + 1.0) < 1.0e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
