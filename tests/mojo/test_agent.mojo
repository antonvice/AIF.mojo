from std.collections import List
from std.testing import TestSuite, assert_equal, assert_true

from aif_mojo.agent import (
    N_PLANNER_KINDS,
    PLANNER_ACTIVE_INFERENCE,
    PLANNER_DYN_CHANNEL,
    PLANNER_LOOPY_BP,
    PLANNER_LOOPY_VBP,
    PLANNER_NUIJTEN,
    PLANNER_PRECISE_INFO_SEEKING,
    PLANNER_REGION_EXTENDED,
    PLANNER_VBP_CHANNEL,
    dense_categorical_belief_update,
    dense_binary_belief_update,
    dispatch_planner_first_action,
    dispatch_planner_first_action_sparse,
    frozen_lake_agent_reset,
    frozen_lake_agent_step,
    frozen_lake_loopy_bp_step,
    marginalize_goal_by_static,
    minigrid_agent_reset,
    minigrid_dense_agent_step,
    minigrid_dense_probabilistic_belief_update,
    minigrid_sparse_agent_step,
    minigrid_sparse_active_step,
    minigrid_sparse_multimodal_belief_update,
    minigrid_sparse_probabilistic_belief_update,
    planning_observation_slice,
    previous_action_distribution,
    rocksample_agent_reset,
    rocksample_agent_step,
    select_smallest_argmax,
    wumpus_agent_reset,
    wumpus_agent_step,
)
from aif_mojo.frozen_lake import (
    generate_frozen_lake_goal,
    generate_frozen_lake_observation,
    generate_frozen_lake_transition,
)
from aif_mojo.active_inference import (
    precompute_obs_channels,
    precompute_pref_to_x,
)
from aif_mojo.numerics import safe_log
from aif_mojo.minigrid import (
    MINIGRID_N_ACTIONS,
    MINIGRID_N_CELL_TYPES,
    MINIGRID_ORIENTATION_DOWN,
    generate_minigrid_goal,
    generate_minigrid_observation_tensor,
    generate_minigrid_orientation_observation_tensor,
    generate_minigrid_transition_indices,
    generate_minigrid_transition_tensor,
    get_fov,
    get_valid_static_configs,
    observation_to_onehot,
)
from aif_mojo.sparse_messages import compute_log_base_sparse
from aif_mojo.rocksample import (
    generate_rocksample_goal,
    generate_rocksample_observation,
    generate_rocksample_quality_configs,
    generate_rocksample_transition,
)
from aif_mojo.wumpus_world import (
    generate_wumpus_goal,
    generate_wumpus_observation,
    generate_wumpus_transition,
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


def single(value: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(value)
    return result^


def log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32]()
    for value in values:
        result.append(safe_log(value))
    return result^


def dispatcher_transition() -> List[Float32]:
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


def single_static_transition() -> List[Float32]:
    # T(new, old, theta=0, action): action 0 stays, action 1 swaps.
    var result = List[Float32]()
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(1.0)
    result.append(1.0)
    result.append(0.0)
    return result^


def dispatcher_observation() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def dispatcher_goal() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.15, 0.75))
    result.extend(pair(0.85, 0.25))
    return result^


def empty_float_list() -> List[Float32]:
    return List[Float32]()


def single_minigrid_config() -> List[Int]:
    var result = List[Int]()
    result.append(0)
    result.append(0)
    return result^


def uniform_orientation_observation() -> List[Float32]:
    var result = List[Float32]()
    for _ in range(4):
        result.append(0.25)
    return result^


def patterned_single_static_goal(n_states: Int) -> List[Float32]:
    """Compact deterministic C(x, theta=1) used by the JAX routing fixture."""
    var result = List[Float32]()
    var total = Float32(0.0)
    for state_idx in range(n_states):
        var base = Float32((state_idx + 4) % 7 + 1)
        var value = base * base * base * base
        value *= value
        result.append(value)
        total += value
    for state_idx in range(n_states):
        result[state_idx] /= total
    return result^


def holes() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    return result^


def start_state() -> List[Float32]:
    var result = List[Float32]()
    result.append(1.0)
    for _ in range(7):
        result.append(0.0)
    return result^


def uniform_static() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.5)
    result.append(0.5)
    return result^


def observation_before_scan() -> List[Float32]:
    # Position state 0, then noisy hole map for theta 0.
    var result = List[Float32]()
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    return result^


def observation_after_scan() -> List[Float32]:
    # Position state 4 (scanned state at position 0), same hole map.
    var result = List[Float32]()
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    return result^


def wumpus_pits() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.0, 1.0))
    result.extend(pair(0.0, 0.0))
    result.extend(pair(0.0, 0.0))
    result.extend(pair(1.0, 0.0))
    return result^


def wumpus_monsters() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.0, 0.0))
    result.extend(pair(1.0, 0.0))
    result.extend(pair(0.0, 0.0))
    result.extend(pair(0.0, 1.0))
    return result^


def wumpus_gold() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.0, 0.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(0.0, 1.0))
    result.extend(pair(0.0, 0.0))
    return result^


def wumpus_observation_fixture() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(0.0)
    result.append(0.0)
    return result^


def rock_positions() -> List[Int]:
    var result = List[Int]()
    result.append(0)
    return result^


def rocksample_observation_fixture() -> List[Float32]:
    # Position 2 is observed; the rock-quality channel reports NO_INFO=2.
    var result = List[Float32]()
    result.append(0.0)
    result.append(0.0)
    result.append(1.0)
    result.append(0.0)
    result.append(2.0)
    return result^


def test_previous_action_is_uniform_then_one_hot() raises:
    var initial = previous_action_distribution(-1, 5)
    for value in initial:
        assert_close(value, 0.2)
    var scan = previous_action_distribution(4, 5)
    for action_idx in range(5):
        if action_idx == 4:
            assert_close(scan[action_idx], 1.0)
        else:
            assert_close(scan[action_idx], 0.0)
    var tied = List[Float32]()
    tied.append(0.5)
    tied.append(0.5)
    tied.append(0.25)
    assert_equal(select_smallest_argmax(tied), 0)


def test_dispatcher_covers_all_eight_planners_and_precomputed_active() raises:
    assert_equal(N_PLANNER_KINDS, 8)
    var expected = List[Float32]()
    expected.extend(pair(0.59064990, 0.40935016))
    expected.extend(pair(0.34874061, 0.65125942))
    expected.extend(pair(0.59449089, 0.40550914))
    expected.extend(pair(0.63885731, 0.36114272))
    expected.extend(pair(0.52000719, 0.47999284))
    expected.extend(pair(0.64047801, 0.35952204))
    expected.extend(pair(0.64002913, 0.35997090))
    expected.extend(pair(0.63661110, 0.36338890))
    for planner_kind in range(N_PLANNER_KINDS):
        var result = dispatch_planner_first_action(
            planner_kind,
            pair(0.65, 0.35),
            pair(0.55, 0.45),
            dispatcher_transition(),
            dispatcher_observation(),
            dispatcher_goal(),
            pair(0.6, 0.4),
            empty_float_list(),
            empty_float_list(),
            2,
            2,
            0.5,
            2,
            2,
            2,
            1,
            2,
            True,
        )
        assert_equal(len(result), 2)
        assert_close(result[0], expected[planner_kind * 2])
        assert_close(result[1], expected[planner_kind * 2 + 1])

    var log_base = List[Float32]()
    log_base.extend(pair(-0.3930426, -0.8556661))
    log_base.extend(pair(-1.1239302, -0.55338514))
    log_base.extend(pair(-0.8324092, -0.7236063))
    log_base.extend(pair(-0.5709296, -0.66358846))
    var local = pair(-1.3985868, -1.4313766)
    var optimized = dispatch_planner_first_action(
        PLANNER_ACTIVE_INFERENCE,
        pair(0.65, 0.35),
        pair(0.55, 0.45),
        dispatcher_transition(),
        dispatcher_observation(),
        dispatcher_goal(),
        pair(0.6, 0.4),
        log_base,
        local,
        2,
        2,
        0.5,
        2,
        2,
        2,
        1,
        2,
        True,
    )
    assert_close(optimized[0], 0.64426464)
    assert_close(optimized[1], 0.35573533)


def test_explicit_goal_kind_disambiguates_single_static_state() raises:
    # Flattening loses JAX rank: C(x) and C(x, theta=1) both have length S.
    # The explicit flag preserves terminal-only versus per-step preference.
    var terminal = dispatch_planner_first_action(
        PLANNER_LOOPY_BP,
        pair(0.8, 0.2),
        single(1.0),
        single_static_transition(),
        empty_float_list(),
        pair(0.1, 0.9),
        pair(0.5, 0.5),
        empty_float_list(),
        empty_float_list(),
        2,
        2,
        1.0,
        2,
        2,
        1,
        0,
        0,
        False,
    )
    var per_step = dispatch_planner_first_action(
        PLANNER_LOOPY_BP,
        pair(0.8, 0.2),
        single(1.0),
        single_static_transition(),
        empty_float_list(),
        pair(0.1, 0.9),
        pair(0.5, 0.5),
        empty_float_list(),
        empty_float_list(),
        2,
        2,
        1.0,
        2,
        2,
        1,
        0,
        0,
        True,
    )
    assert_close(terminal[0], 0.5)
    assert_close(terminal[1], 0.5)
    assert_close(per_step[0], 0.65384614)
    assert_close(per_step[1], 0.34615386)


def test_two_step_frozen_lake_tape_matches_jax_infer_state() raises:
    var transition = generate_frozen_lake_transition(2, holes(), 2, 0.0, 3)
    var observation = generate_frozen_lake_observation(
        2, holes(), 2, 0.05, 0.15
    )
    var step_one = dense_binary_belief_update(
        start_state(),
        uniform_static(),
        transition,
        observation,
        observation_before_scan(),
        -1,
        8,
        2,
        5,
        12,
    )
    assert_equal(len(step_one), 10)
    assert_close(step_one[0], 0.9999982119)
    assert_close(step_one[1], 4.9744136e-7)
    assert_close(step_one[2], 4.9744136e-7)
    assert_close(step_one[4], 8.4042125e-7)
    assert_close(step_one[8], 0.9800000191)
    assert_close(step_one[9], 0.0199999735)

    var q_state = List[Float32]()
    for state_idx in range(8):
        q_state.append(step_one[state_idx])
    var q_static = List[Float32]()
    q_static.append(step_one[8])
    q_static.append(step_one[9])
    var step_two = dense_binary_belief_update(
        q_state,
        q_static,
        transition,
        observation,
        observation_after_scan(),
        4,
        8,
        2,
        5,
        12,
    )
    assert_close(step_two[4], 1.0)
    assert_close(step_two[8], 1.0)
    assert_close(step_two[9], 2.0449001e-8)
    var state_total = Float32(0.0)
    for state_idx in range(8):
        state_total += step_two[state_idx]
    assert_close(state_total, 1.0)


def test_frozen_lake_step_updates_beliefs_and_recedes_horizon() raises:
    var transition = generate_frozen_lake_transition(2, holes(), 2, 0.0, 3)
    var observation = generate_frozen_lake_observation(
        2, holes(), 2, 0.05, 0.15
    )
    var action_prior = List[Float32]()
    for _ in range(5):
        action_prior.append(0.2)
    var result = frozen_lake_loopy_bp_step(
        start_state(),
        uniform_static(),
        transition,
        observation,
        observation_before_scan(),
        generate_frozen_lake_goal(2, holes(), 2, 3, 1.0, 1.0, 1.0),
        action_prior,
        -1,
        2,
        3,
        2,
        8,
        2,
        5,
        12,
    )
    assert_equal(len(result), 12)
    # JAX chooses DOWN for horizon 2; the configured horizon is clipped to 2.
    assert_close(result[0], 1.0)
    assert_close(result[1], 2.0)
    assert_close(result[2], 0.9999982119)
    assert_close(result[10], 0.9800000191)
    assert_close(result[11], 0.0199999735)


def test_frozen_lake_generic_step_dispatches_all_eight_and_goal_kinds() raises:
    var reset = frozen_lake_agent_reset(8, 2)
    assert_equal(len(reset), 10)
    assert_close(reset[0], 1.0)
    assert_close(reset[8], 0.5)
    assert_close(reset[9], 0.5)
    var transition = generate_frozen_lake_transition(2, holes(), 2, 0.0, 3)
    var observation = generate_frozen_lake_observation(
        2, holes(), 2, 0.05, 0.15
    )
    var goal = generate_frozen_lake_goal(2, holes(), 2, 3, 1.0, 1.0, 1.0)
    var expected_actions = List[Int]()
    expected_actions.append(1)
    expected_actions.append(1)
    expected_actions.append(4)
    expected_actions.append(1)
    expected_actions.append(4)
    expected_actions.append(1)
    expected_actions.append(4)
    expected_actions.append(4)
    var first = List[Float32]()
    for planner_kind in range(N_PLANNER_KINDS):
        var result = frozen_lake_agent_step(
            planner_kind,
            start_state(),
            uniform_static(),
            transition,
            observation,
            observation_before_scan(),
            goal,
            previous_action_distribution(-1, 5),
            -1,
            2,
            3,
            2,
            0.5,
            8,
            2,
            5,
            12,
            True,
        )
        assert_close(result[0], Float32(expected_actions[planner_kind]))
        assert_close(result[1], 2.0)
        assert_close(result[2], 0.9999982119)
        assert_close(result[10], 0.9800000191)
        assert_close(result[11], 0.0199999735)
        if planner_kind == 0:
            first = result^

    # Explicitly select terminal C(x), rather than inferring from flat length.
    var posterior_static = pair(first[10], first[11])
    var terminal_goal = marginalize_goal_by_static(goal, posterior_static, 8, 2)
    var terminal = frozen_lake_agent_step(
        PLANNER_LOOPY_BP,
        start_state(),
        uniform_static(),
        transition,
        observation,
        observation_before_scan(),
        terminal_goal,
        previous_action_distribution(-1, 5),
        -1,
        2,
        3,
        2,
        0.5,
        8,
        2,
        5,
        12,
        False,
    )
    assert_close(terminal[0], 1.0)


def test_wumpus_reset_and_step_match_jax_tape() raises:
    var reset = wumpus_agent_reset(8, 2)
    assert_equal(len(reset), 10)
    assert_close(reset[0], 1.0)
    for state_idx in range(1, 8):
        assert_close(reset[state_idx], 0.0)
    assert_close(reset[8], 0.5)
    assert_close(reset[9], 0.5)

    var q_state = List[Float32]()
    for state_idx in range(8):
        q_state.append(reset[state_idx])
    var q_static = List[Float32]()
    q_static.append(reset[8])
    q_static.append(reset[9])
    var transition = generate_wumpus_transition(
        2, wumpus_pits(), wumpus_monsters(), 2, 0.0
    )
    var observation = generate_wumpus_observation(
        2,
        wumpus_pits(),
        wumpus_monsters(),
        wumpus_gold(),
        2,
        0.1,
        0.1,
    )
    var result = wumpus_agent_step(
        PLANNER_LOOPY_BP,
        q_state,
        q_static,
        transition,
        observation,
        wumpus_observation_fixture(),
        generate_wumpus_goal(
            2,
            wumpus_pits(),
            wumpus_monsters(),
            wumpus_gold(),
            2,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        previous_action_distribution(-1, 5),
        -1,
        2,
        3,
        1,
        1.0,
        8,
        2,
        5,
        7,
        2,
        True,
    )
    assert_equal(len(result), 12)
    assert_close(result[0], 2.0)
    assert_close(result[1], 2.0)
    assert_close(result[2], 0.82173163)
    assert_close(result[3], 0.0004611288)
    assert_close(result[6], 0.17734614)
    assert_close(result[10], 0.42759722)
    assert_close(result[11], 0.57240278)


def test_rocksample_categorical_reset_slice_goal_and_two_step_tape() raises:
    var qualities = generate_rocksample_quality_configs(1)
    var transition = generate_rocksample_transition(2, rock_positions(), 1, 0.0)
    var observation = generate_rocksample_observation(
        2, rock_positions(), qualities, 1, 2.0, 0.1
    )
    var goal = generate_rocksample_goal(
        2, rock_positions(), qualities, 1, 2.0, 4.0, 2.0, 1.0
    )
    var reset = rocksample_agent_reset(2, 24, 2)
    assert_equal(len(reset), 26)
    assert_close(reset[2], 1.0)
    assert_close(reset[24], 0.5)
    assert_close(reset[25], 0.5)

    var planning_observation = planning_observation_slice(
        observation, 4, 1, 5, 3, 24, 2
    )
    assert_equal(len(planning_observation), 144)
    var channel_size = 3 * 24 * 2
    for index in range(channel_size):
        assert_close(
            planning_observation[index], observation[4 * channel_size + index]
        )

    var q_state = List[Float32]()
    for state_idx in range(24):
        q_state.append(reset[state_idx])
    var q_static = List[Float32]()
    q_static.append(reset[24])
    q_static.append(reset[25])
    var first = rocksample_agent_step(
        PLANNER_LOOPY_VBP,
        q_state,
        q_static,
        transition,
        observation,
        rocksample_observation_fixture(),
        goal,
        previous_action_distribution(-1, 6),
        -1,
        2,
        3,
        1,
        1.0,
        True,
        24,
        2,
        6,
        4,
        1,
        3,
    )
    assert_equal(len(first), 28)
    assert_close(first[0], 0.0)
    assert_close(first[1], 2.0)
    assert_close(first[2], 0.00037383154)
    assert_close(first[4], 0.99925238)
    assert_close(first[26], 0.5)
    assert_close(first[27], 0.5)

    var terminal_goal = marginalize_goal_by_static(
        goal, pair(first[26], first[27]), 24, 2
    )
    assert_equal(len(terminal_goal), 24)
    assert_close(terminal_goal[0], 0.010939016)
    assert_close(terminal_goal[1], 0.080829062)
    assert_close(terminal_goal[4], 0.0089281304)

    var first_state = List[Float32]()
    for state_idx in range(24):
        first_state.append(first[2 + state_idx])
    var first_static = pair(first[26], first[27])
    var second = rocksample_agent_step(
        PLANNER_LOOPY_VBP,
        first_state,
        first_static,
        transition,
        observation,
        rocksample_observation_fixture(),
        goal,
        previous_action_distribution(0, 6),
        0,
        1,
        3,
        1,
        1.0,
        True,
        24,
        2,
        6,
        4,
        1,
        3,
    )
    assert_close(second[0], 2.0)
    assert_close(second[1], 1.0)
    assert_close(second[2], 4.198775e-7)
    assert_close(second[4], 0.99999905)
    assert_close(second[26], 0.5)
    assert_close(second[27], 0.5)


def test_rocksample_loopy_bp_honors_terminal_goal_only() raises:
    # Frozen JAX agents/rocksample_agent.py:217-222 uniquely bypasses
    # _planning_goal; the other seven agents honor terminal_goal_only. This
    # regression intentionally applies the documented behavior to Loopy BP.
    var qualities = generate_rocksample_quality_configs(1)
    var transition = generate_rocksample_transition(2, rock_positions(), 1, 0.0)
    var goal = generate_rocksample_goal(
        2, rock_positions(), qualities, 1, 2.0, 4.0, 2.0, 1.0
    )
    var q_state = List[Float32]()
    for state_idx in range(24):
        if state_idx == 0:
            q_state.append(1.0)
        else:
            q_state.append(0.0)
    var q_static = pair(0.2, 0.8)
    var terminal_goal = marginalize_goal_by_static(goal, q_static, 24, 2)
    var terminal_policy = dispatch_planner_first_action(
        PLANNER_LOOPY_BP,
        q_state,
        q_static,
        transition,
        empty_float_list(),
        terminal_goal,
        previous_action_distribution(-1, 6),
        empty_float_list(),
        empty_float_list(),
        4,
        1,
        1.0,
        24,
        6,
        2,
        0,
        3,
        False,
    )
    assert_close(terminal_policy[0], 0.1372439)
    assert_close(terminal_policy[1], 0.11140621)
    assert_close(terminal_policy[2], 0.21550426)
    assert_close(terminal_policy[5], 0.2613578)

    var step = rocksample_agent_step(
        PLANNER_LOOPY_BP,
        q_state,
        q_static,
        transition,
        empty_float_list(),
        empty_float_list(),
        goal,
        previous_action_distribution(-1, 6),
        0,
        4,
        4,
        1,
        1.0,
        True,
        24,
        2,
        6,
        0,
        0,
        3,
    )
    assert_close(step[0], 5.0)
    assert_close(step[1], 4.0)
    assert_close(step[2], 1.0)
    assert_close(step[26], 0.2)
    assert_close(step[27], 0.8)


def test_sparse_dispatcher_explicit_goal_kind_with_one_static_state() raises:
    # T(old, action, theta=0) -> new: action 0 stays, action 1 swaps.
    var transition_indices = List[Int]()
    transition_indices.append(0)
    transition_indices.append(1)
    transition_indices.append(1)
    transition_indices.append(0)
    var terminal = dispatch_planner_first_action_sparse(
        PLANNER_LOOPY_BP,
        pair(0.8, 0.2),
        single(1.0),
        transition_indices,
        empty_float_list(),
        pair(0.1, 0.9),
        pair(0.5, 0.5),
        empty_float_list(),
        empty_float_list(),
        2,
        2,
        1.0,
        2,
        2,
        1,
        0,
        0,
        False,
    )
    var per_step = dispatch_planner_first_action_sparse(
        PLANNER_LOOPY_BP,
        pair(0.8, 0.2),
        single(1.0),
        transition_indices,
        empty_float_list(),
        pair(0.1, 0.9),
        pair(0.5, 0.5),
        empty_float_list(),
        empty_float_list(),
        2,
        2,
        1.0,
        2,
        2,
        1,
        0,
        0,
        True,
    )
    assert_close(terminal[0], 0.5)
    assert_close(terminal[1], 0.5)
    assert_close(per_step[0], 0.65384614)
    assert_close(per_step[1], 0.34615386)


def test_minigrid_probabilistic_dense_and_sparse_dispatch_match_jax() raises:
    # Frozen JAX fixture: grid=3, config=(key_pos=0, door_pos=0), FOV=3,
    # previous action TURN_LEFT, exact visual evidence, uniform orientation
    # evidence, horizon=2, one inference/planning iteration, damping=0.5.
    var configs = single_minigrid_config()
    var transition_indices = generate_minigrid_transition_indices(3, configs)
    var transition = generate_minigrid_transition_tensor(3, configs)
    var observation = generate_minigrid_observation_tensor(3, configs, 3)
    var orientation = generate_minigrid_orientation_observation_tensor(3)
    var image = get_fov(0, 0, MINIGRID_ORIENTATION_DOWN, 0, 0, 1, 0, 0, 3, 3)
    var vision = observation_to_onehot(image, 3, 3)
    var orientation_evidence = uniform_orientation_observation()
    var reset = minigrid_agent_reset(3, 1)
    var q_state = List[Float32]()
    for state_idx in range(108):
        q_state.append(reset[state_idx])
    var q_static = single(reset[108])
    var sparse_posterior = minigrid_sparse_probabilistic_belief_update(
        q_state,
        q_static,
        transition_indices,
        observation,
        orientation,
        vision,
        orientation_evidence,
        0,
        1,
        3,
        3,
        1,
    )
    var dense_posterior = minigrid_dense_probabilistic_belief_update(
        q_state,
        q_static,
        transition,
        observation,
        orientation,
        vision,
        orientation_evidence,
        0,
        1,
        3,
        3,
        1,
    )
    assert_equal(len(sparse_posterior), 109)
    assert_close(sparse_posterior[3], 1.0)
    assert_close(sparse_posterior[108], 1.0)
    for index in range(109):
        assert_close(dense_posterior[index], sparse_posterior[index])

    var posterior_state = List[Float32]()
    for state_idx in range(108):
        posterior_state.append(sparse_posterior[state_idx])
    var posterior_static = single(sparse_posterior[108])
    var goal = generate_minigrid_goal(3, 2, 2)
    var action_prior = previous_action_distribution(-1, MINIGRID_N_ACTIONS)
    var log_prior_static = log_values(posterior_static)
    var log_base = compute_log_base_sparse(
        transition_indices,
        log_prior_static,
        108,
        MINIGRID_N_ACTIONS,
        1,
    )
    var log_local = precompute_obs_channels(
        log_values(observation),
        log_prior_static,
        1,
        0.5,
        108,
        1,
        9,
        MINIGRID_N_CELL_TYPES,
    )
    var sparse_kinds = List[Int]()
    sparse_kinds.append(PLANNER_LOOPY_BP)
    sparse_kinds.append(PLANNER_DYN_CHANNEL)
    sparse_kinds.append(PLANNER_NUIJTEN)
    sparse_kinds.append(PLANNER_VBP_CHANNEL)
    sparse_kinds.append(PLANNER_ACTIVE_INFERENCE)
    var sparse_first = List[Float32]()
    sparse_first.append(0.18421066)
    sparse_first.append(0.18420979)
    sparse_first.append(0.18421066)
    sparse_first.append(0.18421066)
    sparse_first.append(0.17533819)
    var sparse_other = List[Float32]()
    sparse_other.append(0.15789466)
    sparse_other.append(0.15789512)
    sparse_other.append(0.15789466)
    sparse_other.append(0.15789466)
    sparse_other.append(0.16233091)
    for kind_idx in range(5):
        var policy = dispatch_planner_first_action_sparse(
            sparse_kinds[kind_idx],
            posterior_state,
            posterior_static,
            transition_indices,
            observation,
            goal,
            action_prior,
            log_base,
            log_local,
            2,
            1,
            0.5,
            108,
            MINIGRID_N_ACTIONS,
            1,
            9,
            MINIGRID_N_CELL_TYPES,
            False,
        )
        for action_idx in range(MINIGRID_N_ACTIONS):
            if action_idx < 2:
                assert_close(policy[action_idx], sparse_first[kind_idx])
            elif action_idx == 2:
                assert_close(policy[action_idx], 0.0)
            else:
                assert_close(policy[action_idx], sparse_other[kind_idx])

    # With K=1, C(x) and C(x, theta) have the same flat length. This patterned
    # JAX fixture makes the routes observable: terminal selects action 0,
    # exactly one preference message selects action 2, and accidental double
    # insertion returns to action 0.
    var theta_goal = patterned_single_static_goal(108)
    var log_preference = precompute_pref_to_x(
        log_values(theta_goal), log_prior_static, 108, 1
    )
    var theta_local = log_local.copy()
    var doubled_local = log_local.copy()
    for state_idx in range(108):
        theta_local[state_idx] += log_preference[state_idx]
        doubled_local[state_idx] += 2.0 * log_preference[state_idx]
    var terminal_policy = dispatch_planner_first_action_sparse(
        PLANNER_ACTIVE_INFERENCE,
        posterior_state,
        posterior_static,
        transition_indices,
        observation,
        theta_goal,
        action_prior,
        log_base,
        log_local,
        2,
        1,
        0.5,
        108,
        MINIGRID_N_ACTIONS,
        1,
        9,
        MINIGRID_N_CELL_TYPES,
        False,
    )
    var theta_policy = dispatch_planner_first_action_sparse(
        PLANNER_ACTIVE_INFERENCE,
        posterior_state,
        posterior_static,
        transition_indices,
        observation,
        theta_goal,
        action_prior,
        log_base,
        theta_local,
        2,
        1,
        0.5,
        108,
        MINIGRID_N_ACTIONS,
        1,
        9,
        MINIGRID_N_CELL_TYPES,
        True,
    )
    var doubled_policy = dispatch_planner_first_action_sparse(
        PLANNER_ACTIVE_INFERENCE,
        posterior_state,
        posterior_static,
        transition_indices,
        observation,
        theta_goal,
        action_prior,
        log_base,
        doubled_local,
        2,
        1,
        0.5,
        108,
        MINIGRID_N_ACTIONS,
        1,
        9,
        MINIGRID_N_CELL_TYPES,
        True,
    )
    assert_close(terminal_policy[0], 0.20281097)
    assert_close(terminal_policy[1], 0.18019331)
    assert_close(terminal_policy[2], 0.19024004)
    assert_close(terminal_policy[3], 0.10668893)
    assert_close(theta_policy[0], 0.19299221)
    assert_close(theta_policy[1], 0.028768688)
    assert_close(theta_policy[2], 0.77823794)
    assert_close(theta_policy[3], 2.8499147e-7)
    assert_close(doubled_policy[0], 0.67232251)
    assert_close(doubled_policy[1], 0.01859572)
    assert_close(doubled_policy[2], 0.30908009)
    assert_close(doubled_policy[3], 4.3399993e-7)
    assert_equal(select_smallest_argmax(terminal_policy), 0)
    assert_equal(select_smallest_argmax(theta_policy), 2)
    assert_equal(select_smallest_argmax(doubled_policy), 0)
    var theta_step = minigrid_sparse_agent_step(
        PLANNER_ACTIVE_INFERENCE,
        q_state,
        q_static,
        transition_indices,
        observation,
        orientation,
        vision,
        orientation_evidence,
        theta_goal,
        action_prior,
        0,
        2,
        2,
        1,
        1,
        0.5,
        3,
        3,
        1,
        True,
    )
    assert_close(theta_step[0], 2.0)
    assert_close(theta_step[1], 2.0)
    assert_close(theta_step[2 + 3], 1.0)

    var sparse_step = minigrid_sparse_agent_step(
        PLANNER_DYN_CHANNEL,
        q_state,
        q_static,
        transition_indices,
        observation,
        orientation,
        vision,
        orientation_evidence,
        goal,
        action_prior,
        0,
        2,
        2,
        1,
        1,
        0.5,
        3,
        3,
        1,
        False,
    )
    assert_close(sparse_step[0], 0.0)
    assert_close(sparse_step[1], 2.0)
    assert_close(sparse_step[2 + 3], 1.0)
    var dense_step = minigrid_dense_agent_step(
        PLANNER_LOOPY_BP,
        q_state,
        q_static,
        transition,
        observation,
        orientation,
        vision,
        orientation_evidence,
        goal,
        action_prior,
        0,
        2,
        2,
        1,
        1,
        0.5,
        3,
        3,
        1,
        False,
    )
    assert_close(dense_step[0], 0.0)
    assert_close(dense_step[1], 2.0)
    assert_close(dense_step[2 + 3], 1.0)


def test_large_minigrid_sparse_multimodal_active_tape_matches_jax() raises:
    # Full 4x4 configuration family: S=192, theta=48. No dense transition is
    # constructed; inference and planning both use deterministic sparse indices.
    var configs = get_valid_static_configs(4)
    assert_equal(len(configs), 96)
    var transition_indices = generate_minigrid_transition_indices(4, configs)
    assert_equal(len(transition_indices), 192 * MINIGRID_N_ACTIONS * 48)
    var observation = generate_minigrid_observation_tensor(4, configs, 3)
    assert_equal(len(observation), 3 * 3 * MINIGRID_N_CELL_TYPES * 192 * 48)
    var orientation = generate_minigrid_orientation_observation_tensor(4)

    # JAX config 0=(key_pos 0, door_pos 0), state observation at position
    # (0,0), orientation DOWN, key on ground.
    var image = get_fov(0, 0, MINIGRID_ORIENTATION_DOWN, 0, 0, 1, 0, 0, 4, 3)
    var expected_image = List[Int]()
    expected_image.append(2)
    expected_image.append(2)
    expected_image.append(4)
    expected_image.append(1)
    expected_image.append(1)
    expected_image.append(5)
    expected_image.append(2)
    expected_image.append(2)
    expected_image.append(2)
    for index in range(9):
        assert_equal(image[index], expected_image[index])

    var reset = minigrid_agent_reset(4, 48)
    assert_equal(len(reset), 240)
    assert_close(reset[0], 0.03125)
    assert_close(reset[3], 0.03125)
    assert_close(reset[1], 0.0)
    assert_close(reset[191], 0.0)
    assert_close(reset[192], 1.0 / 48.0)

    var q_state = List[Float32]()
    for state_idx in range(192):
        q_state.append(reset[state_idx])
    var q_static = List[Float32]()
    for static_idx in range(48):
        q_static.append(reset[192 + static_idx])
    var posterior = minigrid_sparse_multimodal_belief_update(
        q_state,
        q_static,
        transition_indices,
        observation,
        orientation,
        image,
        MINIGRID_ORIENTATION_DOWN,
        0,
        1,
        4,
        3,
        48,
    )
    assert_equal(len(posterior), 240)
    assert_close(posterior[3], 0.5)
    assert_close(posterior[15], 0.5)
    assert_close(posterior[4], 1.3823989e-8)
    assert_close(posterior[17], 1.3823989e-8)
    assert_close(posterior[192], 0.44444436)
    assert_close(posterior[193], 0.22222228)
    assert_close(posterior[200], 0.22222228)
    assert_close(posterior[201], 0.11111109)

    var step = minigrid_sparse_active_step(
        q_state,
        q_static,
        transition_indices,
        observation,
        orientation,
        image,
        MINIGRID_ORIENTATION_DOWN,
        generate_minigrid_goal(4, 3, 3),
        previous_action_distribution(-1, MINIGRID_N_ACTIONS),
        0,
        1,
        3,
        1,
        1,
        1.0,
        4,
        3,
        48,
        False,
    )
    assert_equal(len(step), 242)
    assert_close(step[0], 0.0)
    assert_close(step[1], 1.0)
    assert_close(step[2 + 3], 0.5)
    assert_close(step[2 + 15], 0.5)
    assert_close(step[2 + 192], 0.44444436)
    assert_close(step[2 + 193], 0.22222228)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
