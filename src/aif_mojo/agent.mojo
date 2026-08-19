from std.collections import List
from std.math import log

from aif_mojo.active_inference import (
    active_inference_planning_dense,
    active_inference_planning_dense_theta_goal,
    active_inference_planning_precomputed,
    active_inference_planning_precomputed_theta_goal,
    precompute_obs_channels,
    precompute_pref_to_x,
)
from aif_mojo.dyn_channel_loopy_bp import (
    dyn_channel_loopy_bp_planning_dense,
    dyn_channel_loopy_bp_planning_dense_theta_goal,
    dyn_channel_loopy_bp_planning_sparse,
    dyn_channel_loopy_bp_planning_sparse_theta_goal,
)
from aif_mojo.loopy_bp import (
    loopy_bp_planning_dense,
    loopy_bp_planning_dense_theta_goal,
    loopy_bp_planning_sparse,
    loopy_bp_planning_sparse_theta_goal,
)
from aif_mojo.loopy_vbp import (
    loopy_vbp_planning_dense,
    loopy_vbp_planning_dense_theta_goal,
)
from aif_mojo.minigrid import (
    MINIGRID_N_ACTIONS,
    MINIGRID_N_CELL_TYPES,
    MINIGRID_N_DOOR_KEY_STATES,
    MINIGRID_N_ORIENTATIONS,
    action_to_onehot,
    direction_to_onehot,
    flatten_state_index,
    observation_to_onehot,
)
from aif_mojo.nuijten_mp import (
    nuijten_mp_planning_dense,
    nuijten_mp_planning_dense_theta_goal,
    nuijten_mp_planning_sparse,
    nuijten_mp_planning_sparse_theta_goal,
)
from aif_mojo.numerics import logsumexp, safe_log, softmax
from aif_mojo.precise_info_seeking import (
    precise_info_seeking_planning_dense,
    precise_info_seeking_planning_dense_theta_goal,
)
from aif_mojo.region_extended_loopy_bp import (
    region_extended_loopy_bp_planning_dense,
    region_extended_loopy_bp_planning_dense_theta_goal,
)
from aif_mojo.sparse_messages import compute_log_base_sparse
from aif_mojo.state_inference import (
    state_inference_step,
    state_inference_step_sparse,
)
from aif_mojo.vbp_channel import (
    vbp_channel_planning_dense,
    vbp_channel_planning_dense_theta_goal,
    vbp_channel_planning_sparse,
    vbp_channel_planning_sparse_theta_goal,
)


comptime BAYES_EPSILON = Float32(1.0e-12)
comptime PLANNER_LOOPY_BP = 0
comptime PLANNER_LOOPY_VBP = 1
comptime PLANNER_REGION_EXTENDED = 2
comptime PLANNER_DYN_CHANNEL = 3
comptime PLANNER_NUIJTEN = 4
comptime PLANNER_VBP_CHANNEL = 5
comptime PLANNER_PRECISE_INFO_SEEKING = 6
comptime PLANNER_ACTIVE_INFERENCE = 7
comptime N_PLANNER_KINDS = 8


def previous_action_distribution(
    previous_action: Int, n_actions: Int
) -> List[Float32]:
    """Return uniform weights before the first step, otherwise a one-hot action.
    """
    debug_assert(n_actions > 0, "action count must be positive")
    debug_assert(
        previous_action >= -1 and previous_action < n_actions,
        "previous action out of range",
    )
    var result = List[Float32]()
    for action_idx in range(n_actions):
        if previous_action < 0:
            result.append(1.0 / Float32(n_actions))
        elif action_idx == previous_action:
            result.append(1.0)
        else:
            result.append(0.0)
    return result^


def dense_categorical_belief_update(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    categorical_observation: List[Float32],
    previous_action: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_channels: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Match the dense categorical Bayesian update used by JAX task agents.

    T is flattened as (new, old, theta, action), B as
    (channel, outcome, state, theta). The result is q(x) followed by
    q(theta).
    """
    debug_assert(
        len(q_current_state) == n_states, "state belief shape mismatch"
    )
    debug_assert(
        len(q_static_state) == n_static, "static belief shape mismatch"
    )
    debug_assert(
        len(transition_tensor) == n_states * n_states * n_static * n_actions,
        "transition tensor shape mismatch",
    )
    debug_assert(
        len(observation_tensor)
        == n_channels * n_obs_types * n_states * n_static,
        "observation tensor shape mismatch",
    )
    debug_assert(
        len(categorical_observation) == n_channels,
        "categorical observation shape mismatch",
    )

    var action_weights = previous_action_distribution(
        previous_action, n_actions
    )
    var predicted = List[Float32]()
    var predicted_total = Float32(0.0)
    for new_idx in range(n_states):
        var value = Float32(0.0)
        for old_idx in range(n_states):
            for static_idx in range(n_static):
                for action_idx in range(n_actions):
                    var transition_offset = (
                        (new_idx * n_states + old_idx) * n_static + static_idx
                    ) * n_actions + action_idx
                    value += (
                        transition_tensor[transition_offset]
                        * q_current_state[old_idx]
                        * q_static_state[static_idx]
                        * action_weights[action_idx]
                    )
        predicted.append(value)
        predicted_total += value
    for state_idx in range(n_states):
        predicted[state_idx] /= predicted_total + BAYES_EPSILON

    var log_joint = List[Float32]()
    for state_idx in range(n_states):
        for static_idx in range(n_static):
            var log_likelihood = Float32(0.0)
            for channel_idx in range(n_channels):
                var outcome_idx = Int(
                    categorical_observation[channel_idx] + 0.5
                )
                debug_assert(
                    outcome_idx >= 0 and outcome_idx < n_obs_types,
                    "observation outcome out of range",
                )
                var observation_offset = (
                    (channel_idx * n_obs_types + outcome_idx) * n_states
                    + state_idx
                ) * n_static + static_idx
                log_likelihood += log(
                    observation_tensor[observation_offset] + BAYES_EPSILON
                )
            log_joint.append(
                log_likelihood
                + log(predicted[state_idx] + BAYES_EPSILON)
                + log(q_static_state[static_idx] + BAYES_EPSILON)
            )

    var state_logits = List[Float32]()
    for state_idx in range(n_states):
        var terms = List[Float32]()
        for static_idx in range(n_static):
            terms.append(log_joint[state_idx * n_static + static_idx])
        state_logits.append(logsumexp(terms))
    var q_new_state = softmax(state_logits)

    var static_logits = List[Float32]()
    for static_idx in range(n_static):
        var terms = List[Float32]()
        for state_idx in range(n_states):
            terms.append(log_joint[state_idx * n_static + static_idx])
        static_logits.append(logsumexp(terms))
    var q_new_static = softmax(static_logits)

    var result = List[Float32]()
    result.extend(q_new_state^)
    result.extend(q_new_static^)
    return result^


def dense_binary_belief_update(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    binary_observation: List[Float32],
    previous_action: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_channels: Int,
) -> List[Float32]:
    """Backward-compatible binary specialization."""
    return dense_categorical_belief_update(
        q_current_state,
        q_static_state,
        transition_tensor,
        observation_tensor,
        binary_observation,
        previous_action,
        n_states,
        n_static,
        n_actions,
        n_channels,
        2,
    )


def _first_action_distribution(
    planner_result: List[Float32], n_actions: Int
) -> List[Float32]:
    debug_assert(len(planner_result) >= n_actions, "planner result too short")
    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(planner_result[action_idx])
    return result^


def select_smallest_argmax(action_distribution: List[Float32]) -> Int:
    """Return the first maximum, matching jnp.argmax tie behavior."""
    debug_assert(len(action_distribution) > 0, "empty action distribution")
    var selected = Int(0)
    for action_idx in range(1, len(action_distribution)):
        if action_distribution[action_idx] > action_distribution[selected]:
            selected = action_idx
    return selected


def marginalize_goal_by_static(
    goal_by_static: List[Float32],
    q_static_state: List[Float32],
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Convert C(state,theta) into the terminal state goal used by JAX."""
    debug_assert(
        len(goal_by_static) == n_states * n_static,
        "theta goal shape mismatch",
    )
    debug_assert(len(q_static_state) == n_static, "static belief mismatch")
    var log_preferences = List[Float32]()
    for state_idx in range(n_states):
        var terms = List[Float32]()
        for static_idx in range(n_static):
            terms.append(
                safe_log(goal_by_static[state_idx * n_static + static_idx])
                + safe_log(q_static_state[static_idx])
            )
        log_preferences.append(logsumexp(terms))
    return softmax(log_preferences)


def planning_observation_slice(
    observation_tensor: List[Float32],
    first_channel: Int,
    channel_count: Int,
    n_channels: Int,
    n_obs_types: Int,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Select contiguous planning channels from B(channel,obs,state,theta)."""
    debug_assert(
        len(observation_tensor)
        == n_channels * n_obs_types * n_states * n_static,
        "observation tensor shape mismatch",
    )
    debug_assert(
        first_channel >= 0
        and channel_count >= 0
        and first_channel + channel_count <= n_channels,
        "planning observation slice out of range",
    )
    var channel_size = n_obs_types * n_states * n_static
    var result = List[Float32]()
    for index in range(channel_count * channel_size):
        result.append(observation_tensor[first_channel * channel_size + index])
    return result^


def dispatch_planner_first_action(
    planner_kind: Int,
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    planning_observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    log_base: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_planning_channels: Int,
    n_obs_types: Int,
    theta_goal: Bool,
) -> List[Float32]:
    """Dispatch all retained planners and return only the first action."""
    debug_assert(
        planner_kind >= 0 and planner_kind < N_PLANNER_KINDS,
        "unknown planner kind",
    )
    if planner_kind == PLANNER_LOOPY_BP:
        if theta_goal:
            return _first_action_distribution(
                loopy_bp_planning_dense_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    n_states,
                    n_actions,
                    n_static,
                ),
                n_actions,
            )
        return _first_action_distribution(
            loopy_bp_planning_dense(
                q_current_state,
                q_static_state,
                transition_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                n_states,
                n_actions,
                n_static,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_LOOPY_VBP:
        if theta_goal:
            return _first_action_distribution(
                loopy_vbp_planning_dense_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_tensor,
                    goal,
                    horizon,
                    n_iterations,
                    n_states,
                    n_actions,
                    n_static,
                ),
                n_actions,
            )
        return _first_action_distribution(
            loopy_vbp_planning_dense(
                q_current_state,
                q_static_state,
                transition_tensor,
                goal,
                horizon,
                n_iterations,
                n_states,
                n_actions,
                n_static,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_REGION_EXTENDED:
        if theta_goal:
            return _first_action_distribution(
                region_extended_loopy_bp_planning_dense_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_tensor,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    damping,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            region_extended_loopy_bp_planning_dense(
                q_current_state,
                q_static_state,
                transition_tensor,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_DYN_CHANNEL:
        if theta_goal:
            return _first_action_distribution(
                dyn_channel_loopy_bp_planning_dense_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_tensor,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    damping,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            dyn_channel_loopy_bp_planning_dense(
                q_current_state,
                q_static_state,
                transition_tensor,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_NUIJTEN:
        if theta_goal:
            return _first_action_distribution(
                nuijten_mp_planning_dense_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_tensor,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            nuijten_mp_planning_dense(
                q_current_state,
                q_static_state,
                transition_tensor,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_VBP_CHANNEL:
        if theta_goal:
            return _first_action_distribution(
                vbp_channel_planning_dense_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_tensor,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    damping,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            vbp_channel_planning_dense(
                q_current_state,
                q_static_state,
                transition_tensor,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_PRECISE_INFO_SEEKING:
        if theta_goal:
            return _first_action_distribution(
                precise_info_seeking_planning_dense_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_tensor,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    damping,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            precise_info_seeking_planning_dense(
                q_current_state,
                q_static_state,
                transition_tensor,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )

    if len(log_base) > 0:
        if theta_goal:
            return _first_action_distribution(
                active_inference_planning_precomputed_theta_goal(
                    q_current_state,
                    q_static_state,
                    log_base,
                    log_local_to_x,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    damping,
                    n_states,
                    n_actions,
                    n_static,
                ),
                n_actions,
            )
        return _first_action_distribution(
            active_inference_planning_precomputed(
                q_current_state,
                q_static_state,
                log_base,
                log_local_to_x,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
            ),
            n_actions,
        )
    if theta_goal:
        return _first_action_distribution(
            active_inference_planning_dense_theta_goal(
                q_current_state,
                q_static_state,
                transition_tensor,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    return _first_action_distribution(
        active_inference_planning_dense(
            q_current_state,
            q_static_state,
            transition_tensor,
            planning_observation_tensor,
            goal,
            action_prior,
            horizon,
            n_iterations,
            damping,
            n_states,
            n_actions,
            n_static,
            n_planning_channels,
            n_obs_types,
        ),
        n_actions,
    )


def dispatch_planner_first_action_sparse(
    planner_kind: Int,
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    planning_observation_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    log_base: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
    n_planning_channels: Int,
    n_obs_types: Int,
    theta_goal: Bool,
) -> List[Float32]:
    """Dispatch the five JAX planners that support deterministic sparse T."""
    debug_assert(
        planner_kind == PLANNER_LOOPY_BP
        or planner_kind == PLANNER_DYN_CHANNEL
        or planner_kind == PLANNER_NUIJTEN
        or planner_kind == PLANNER_VBP_CHANNEL
        or planner_kind == PLANNER_ACTIVE_INFERENCE,
        "planner has no sparse implementation",
    )
    if planner_kind == PLANNER_LOOPY_BP:
        if theta_goal:
            return _first_action_distribution(
                loopy_bp_planning_sparse_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_indices,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    n_states,
                    n_actions,
                    n_static,
                ),
                n_actions,
            )
        return _first_action_distribution(
            loopy_bp_planning_sparse(
                q_current_state,
                q_static_state,
                transition_indices,
                goal,
                action_prior,
                horizon,
                n_iterations,
                n_states,
                n_actions,
                n_static,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_DYN_CHANNEL:
        if theta_goal:
            return _first_action_distribution(
                dyn_channel_loopy_bp_planning_sparse_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_indices,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    damping,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            dyn_channel_loopy_bp_planning_sparse(
                q_current_state,
                q_static_state,
                transition_indices,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_NUIJTEN:
        if theta_goal:
            return _first_action_distribution(
                nuijten_mp_planning_sparse_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_indices,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            nuijten_mp_planning_sparse(
                q_current_state,
                q_static_state,
                transition_indices,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    if planner_kind == PLANNER_VBP_CHANNEL:
        if theta_goal:
            return _first_action_distribution(
                vbp_channel_planning_sparse_theta_goal(
                    q_current_state,
                    q_static_state,
                    transition_indices,
                    planning_observation_tensor,
                    goal,
                    action_prior,
                    horizon,
                    n_iterations,
                    damping,
                    n_states,
                    n_actions,
                    n_static,
                    n_planning_channels,
                    n_obs_types,
                ),
                n_actions,
            )
        return _first_action_distribution(
            vbp_channel_planning_sparse(
                q_current_state,
                q_static_state,
                transition_indices,
                planning_observation_tensor,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
                n_planning_channels,
                n_obs_types,
            ),
            n_actions,
        )
    debug_assert(len(log_base) > 0, "Active sparse log-base is required")
    if theta_goal:
        return _first_action_distribution(
            active_inference_planning_precomputed_theta_goal(
                q_current_state,
                q_static_state,
                log_base,
                log_local_to_x,
                goal,
                action_prior,
                horizon,
                n_iterations,
                damping,
                n_states,
                n_actions,
                n_static,
            ),
            n_actions,
        )
    return _first_action_distribution(
        active_inference_planning_precomputed(
            q_current_state,
            q_static_state,
            log_base,
            log_local_to_x,
            goal,
            action_prior,
            horizon,
            n_iterations,
            damping,
            n_states,
            n_actions,
            n_static,
        ),
        n_actions,
    )


def _reset_beliefs(
    start_state: Int, n_states: Int, n_static: Int
) -> List[Float32]:
    debug_assert(
        start_state >= 0 and start_state < n_states, "start state out of range"
    )
    var result = List[Float32]()
    for state_idx in range(n_states):
        if state_idx == start_state:
            result.append(1.0)
        else:
            result.append(0.0)
    for _ in range(n_static):
        result.append(1.0 / Float32(n_static))
    return result^


def frozen_lake_agent_reset(n_states: Int, n_static: Int) -> List[Float32]:
    """Return the JAX Frozen Lake reset belief."""
    return _reset_beliefs(0, n_states, n_static)


def wumpus_agent_reset(n_states: Int, n_static: Int) -> List[Float32]:
    """Return one-hot state zero followed by a uniform static belief."""
    return _reset_beliefs(0, n_states, n_static)


def rocksample_agent_reset(
    start_state: Int, n_states: Int, n_static: Int
) -> List[Float32]:
    """Return the configured one-hot start followed by uniform qualities."""
    return _reset_beliefs(start_state, n_states, n_static)


def _agent_step_result(
    action_distribution: List[Float32],
    horizon: Int,
    beliefs: List[Float32],
) -> List[Float32]:
    var result = List[Float32]()
    result.append(Float32(select_smallest_argmax(action_distribution)))
    result.append(Float32(horizon))
    for value in beliefs:
        result.append(value)
    return result^


def frozen_lake_agent_step(
    planner_kind: Int,
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    observation: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    previous_action: Int,
    time_remaining: Int,
    planning_horizon: Int,
    planning_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_channels: Int,
    theta_goal: Bool,
) -> List[Float32]:
    """Pure Frozen Lake inference plus any retained dense planner."""
    var beliefs = dense_binary_belief_update(
        q_current_state,
        q_static_state,
        transition_tensor,
        observation_tensor,
        observation,
        previous_action,
        n_states,
        n_static,
        n_actions,
        n_channels,
    )
    var q_new_state = List[Float32]()
    for state_idx in range(n_states):
        q_new_state.append(beliefs[state_idx])
    var q_new_static = List[Float32]()
    for static_idx in range(n_static):
        q_new_static.append(beliefs[n_states + static_idx])
    var n_directional_channels = n_channels - n_states
    debug_assert(
        n_directional_channels >= 0, "invalid Frozen observation shape"
    )
    var planning_observation = planning_observation_slice(
        observation_tensor,
        n_states,
        n_directional_channels,
        n_channels,
        2,
        n_states,
        n_static,
    )
    var horizon = min(time_remaining, planning_horizon)
    debug_assert(horizon > 0, "receding horizon must be positive")
    var action_distribution = dispatch_planner_first_action(
        planner_kind,
        q_new_state,
        q_new_static,
        transition_tensor,
        planning_observation,
        goal,
        action_prior,
        List[Float32](),
        List[Float32](),
        horizon,
        planning_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_directional_channels,
        2,
        theta_goal,
    )
    return _agent_step_result(action_distribution, horizon, beliefs)


def wumpus_agent_step(
    planner_kind: Int,
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    observation: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    previous_action: Int,
    time_remaining: Int,
    planning_horizon: Int,
    planning_iterations: Int,
    damping: Float32,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_channels: Int,
    n_obs_types: Int,
    theta_goal: Bool,
) -> List[Float32]:
    """Pure Wumpus inference and planning step.

    Returns [selected action, receding horizon, q(state)..., q(theta)...].
    """
    var beliefs = dense_categorical_belief_update(
        q_current_state,
        q_static_state,
        transition_tensor,
        observation_tensor,
        observation,
        previous_action,
        n_states,
        n_static,
        n_actions,
        n_channels,
        n_obs_types,
    )
    var q_new_state = List[Float32]()
    for state_idx in range(n_states):
        q_new_state.append(beliefs[state_idx])
    var q_new_static = List[Float32]()
    for static_idx in range(n_static):
        q_new_static.append(beliefs[n_states + static_idx])
    var horizon = min(time_remaining, planning_horizon)
    debug_assert(horizon > 0, "receding horizon must be positive")
    var action_distribution = dispatch_planner_first_action(
        planner_kind,
        q_new_state,
        q_new_static,
        transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        List[Float32](),
        List[Float32](),
        horizon,
        planning_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_channels,
        n_obs_types,
        theta_goal,
    )
    return _agent_step_result(action_distribution, horizon, beliefs)


def rocksample_agent_step(
    planner_kind: Int,
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    observation: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    previous_action: Int,
    time_remaining: Int,
    planning_horizon: Int,
    planning_iterations: Int,
    damping: Float32,
    terminal_goal_only: Bool,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_position_channels: Int,
    n_rock_channels: Int,
    n_obs_types: Int,
) -> List[Float32]:
    """Pure RockSample categorical inference and planning step.

    Full observations update beliefs. Only rock-quality channels participate
    in planning. Returns [action, receding horizon, q(state)..., q(theta)...].
    """
    var n_channels = n_position_channels + n_rock_channels
    var beliefs = dense_categorical_belief_update(
        q_current_state,
        q_static_state,
        transition_tensor,
        observation_tensor,
        observation,
        previous_action,
        n_states,
        n_static,
        n_actions,
        n_channels,
        n_obs_types,
    )
    var q_new_state = List[Float32]()
    for state_idx in range(n_states):
        q_new_state.append(beliefs[state_idx])
    var q_new_static = List[Float32]()
    for static_idx in range(n_static):
        q_new_static.append(beliefs[n_states + static_idx])
    var planning_observation = planning_observation_slice(
        observation_tensor,
        n_position_channels,
        n_rock_channels,
        n_channels,
        n_obs_types,
        n_states,
        n_static,
    )
    var planning_goal = goal_by_static.copy()
    var theta_goal = True
    if terminal_goal_only:
        planning_goal = marginalize_goal_by_static(
            goal_by_static, q_new_static, n_states, n_static
        )
        theta_goal = False
    var horizon = min(time_remaining, planning_horizon)
    debug_assert(horizon > 0, "receding horizon must be positive")
    var action_distribution = dispatch_planner_first_action(
        planner_kind,
        q_new_state,
        q_new_static,
        transition_tensor,
        planning_observation,
        planning_goal,
        action_prior,
        List[Float32](),
        List[Float32](),
        horizon,
        planning_iterations,
        damping,
        n_states,
        n_actions,
        n_static,
        n_rock_channels,
        n_obs_types,
        theta_goal,
    )
    return _agent_step_result(action_distribution, horizon, beliefs)


def minigrid_agent_reset(grid_size: Int, n_static: Int) -> List[Float32]:
    """Match the flat-tensor MiniGrid reset distribution.

    Valid pre-wall locations and all orientations are uniform with the key on
    the ground. The result is q(state) followed by uniform q(theta).
    """
    var n_locations = grid_size * grid_size
    var n_states = (
        n_locations * MINIGRID_N_ORIENTATIONS * MINIGRID_N_DOOR_KEY_STATES
    )
    var n_valid_locations = n_locations - 2 * grid_size
    debug_assert(n_valid_locations > 0, "grid too small")
    debug_assert(n_static > 0, "static count must be positive")
    var result = List[Float32]()
    for _ in range(n_states):
        result.append(0.0)
    var probability = 1.0 / Float32(n_valid_locations * MINIGRID_N_ORIENTATIONS)
    for location in range(n_valid_locations):
        for orientation in range(MINIGRID_N_ORIENTATIONS):
            var state = flatten_state_index(
                location,
                orientation,
                0,
                n_locations,
                MINIGRID_N_ORIENTATIONS,
                MINIGRID_N_DOOR_KEY_STATES,
            )
            result[state] = probability
    for _ in range(n_static):
        result.append(1.0 / Float32(n_static))
    return result^


def minigrid_sparse_probabilistic_belief_update(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    orientation_tensor: List[Float32],
    vision_observation: List[Float32],
    orientation_observation: List[Float32],
    previous_action: Int,
    n_inference_iterations: Int,
    grid_size: Int,
    fov_size: Int,
    n_static: Int,
) -> List[Float32]:
    """Integrate full MiniGrid vision and orientation distributions."""
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    debug_assert(
        len(vision_observation) == fov_size * fov_size * MINIGRID_N_CELL_TYPES,
        "vision observation shape mismatch",
    )
    debug_assert(
        len(orientation_observation) == MINIGRID_N_ORIENTATIONS,
        "orientation observation shape mismatch",
    )
    var action = action_to_onehot(previous_action)
    return state_inference_step_sparse(
        q_current_state,
        q_static_state,
        transition_indices,
        observation_tensor,
        orientation_tensor,
        vision_observation,
        orientation_observation,
        action,
        n_inference_iterations,
        n_states,
        n_static,
        MINIGRID_N_ACTIONS,
        fov_size * fov_size,
        MINIGRID_N_CELL_TYPES,
        MINIGRID_N_ORIENTATIONS,
    )


def minigrid_dense_probabilistic_belief_update(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    orientation_tensor: List[Float32],
    vision_observation: List[Float32],
    orientation_observation: List[Float32],
    previous_action: Int,
    n_inference_iterations: Int,
    grid_size: Int,
    fov_size: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense counterpart of MiniGrid probabilistic belief integration."""
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    debug_assert(
        len(vision_observation) == fov_size * fov_size * MINIGRID_N_CELL_TYPES,
        "vision observation shape mismatch",
    )
    debug_assert(
        len(orientation_observation) == MINIGRID_N_ORIENTATIONS,
        "orientation observation shape mismatch",
    )
    var action = action_to_onehot(previous_action)
    return state_inference_step(
        q_current_state,
        q_static_state,
        transition_tensor,
        observation_tensor,
        orientation_tensor,
        vision_observation,
        orientation_observation,
        action,
        n_inference_iterations,
        n_states,
        n_static,
        MINIGRID_N_ACTIONS,
        fov_size * fov_size,
        MINIGRID_N_CELL_TYPES,
        MINIGRID_N_ORIENTATIONS,
    )


def minigrid_sparse_multimodal_belief_update(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    orientation_tensor: List[Float32],
    vision_image: List[Int],
    direction: Int,
    previous_action: Int,
    n_inference_iterations: Int,
    grid_size: Int,
    fov_size: Int,
    n_static: Int,
) -> List[Float32]:
    """One-hot convenience wrapper for MiniGrid belief integration."""
    return minigrid_sparse_probabilistic_belief_update(
        q_current_state,
        q_static_state,
        transition_indices,
        observation_tensor,
        orientation_tensor,
        observation_to_onehot(vision_image, fov_size, fov_size),
        direction_to_onehot(direction),
        previous_action,
        n_inference_iterations,
        grid_size,
        fov_size,
        n_static,
    )


def minigrid_sparse_agent_step(
    planner_kind: Int,
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    orientation_tensor: List[Float32],
    vision_observation: List[Float32],
    orientation_observation: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    previous_action: Int,
    time_remaining: Int,
    planning_horizon: Int,
    n_inference_iterations: Int,
    n_planning_iterations: Int,
    damping: Float32,
    grid_size: Int,
    fov_size: Int,
    n_static: Int,
    theta_goal: Bool,
) -> List[Float32]:
    """Pure probabilistic MiniGrid step for all five sparse planners.

    Returns [action, receding horizon, q(state)..., q(theta)...].
    """
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    var beliefs = minigrid_sparse_probabilistic_belief_update(
        q_current_state,
        q_static_state,
        transition_indices,
        observation_tensor,
        orientation_tensor,
        vision_observation,
        orientation_observation,
        previous_action,
        n_inference_iterations,
        grid_size,
        fov_size,
        n_static,
    )
    var q_new_state = List[Float32]()
    for state_idx in range(n_states):
        q_new_state.append(beliefs[state_idx])
    var q_new_static = List[Float32]()
    for static_idx in range(n_static):
        q_new_static.append(beliefs[n_states + static_idx])

    var n_fov = fov_size * fov_size
    var log_base = List[Float32]()
    var log_local_to_x = List[Float32]()
    if planner_kind == PLANNER_ACTIVE_INFERENCE:
        var log_prior_theta = List[Float32]()
        for value in q_new_static:
            log_prior_theta.append(safe_log(value))
        log_base = compute_log_base_sparse(
            transition_indices,
            log_prior_theta,
            n_states,
            MINIGRID_N_ACTIONS,
            n_static,
        )
        var log_observation = List[Float32]()
        for value in observation_tensor:
            log_observation.append(safe_log(value))
        log_local_to_x = precompute_obs_channels(
            log_observation,
            log_prior_theta,
            n_planning_iterations,
            damping,
            n_states,
            n_static,
            n_fov,
            MINIGRID_N_CELL_TYPES,
        )
        if theta_goal:
            debug_assert(
                len(goal) == n_states * n_static,
                "MiniGrid theta goal shape mismatch",
            )
            var log_goal = List[Float32]()
            for value in goal:
                log_goal.append(safe_log(value))
            var log_pref_to_x = precompute_pref_to_x(
                log_goal, log_prior_theta, n_states, n_static
            )
            for state_idx in range(n_states):
                log_local_to_x[state_idx] += log_pref_to_x[state_idx]

    var horizon = min(time_remaining, planning_horizon)
    debug_assert(horizon > 0, "receding horizon must be positive")
    var action_distribution = dispatch_planner_first_action_sparse(
        planner_kind,
        q_new_state,
        q_new_static,
        transition_indices,
        observation_tensor,
        goal,
        action_prior,
        log_base,
        log_local_to_x,
        horizon,
        n_planning_iterations,
        damping,
        n_states,
        MINIGRID_N_ACTIONS,
        n_static,
        n_fov,
        MINIGRID_N_CELL_TYPES,
        theta_goal,
    )
    return _agent_step_result(action_distribution, horizon, beliefs)


def minigrid_dense_agent_step(
    planner_kind: Int,
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    orientation_tensor: List[Float32],
    vision_observation: List[Float32],
    orientation_observation: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    previous_action: Int,
    time_remaining: Int,
    planning_horizon: Int,
    n_inference_iterations: Int,
    n_planning_iterations: Int,
    damping: Float32,
    grid_size: Int,
    fov_size: Int,
    n_static: Int,
    theta_goal: Bool,
) -> List[Float32]:
    """Pure probabilistic MiniGrid step for all eight dense planners."""
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    var beliefs = minigrid_dense_probabilistic_belief_update(
        q_current_state,
        q_static_state,
        transition_tensor,
        observation_tensor,
        orientation_tensor,
        vision_observation,
        orientation_observation,
        previous_action,
        n_inference_iterations,
        grid_size,
        fov_size,
        n_static,
    )
    var q_new_state = List[Float32]()
    for state_idx in range(n_states):
        q_new_state.append(beliefs[state_idx])
    var q_new_static = List[Float32]()
    for static_idx in range(n_static):
        q_new_static.append(beliefs[n_states + static_idx])
    var horizon = min(time_remaining, planning_horizon)
    debug_assert(horizon > 0, "receding horizon must be positive")
    var action_distribution = dispatch_planner_first_action(
        planner_kind,
        q_new_state,
        q_new_static,
        transition_tensor,
        observation_tensor,
        goal,
        action_prior,
        List[Float32](),
        List[Float32](),
        horizon,
        n_planning_iterations,
        damping,
        n_states,
        MINIGRID_N_ACTIONS,
        n_static,
        fov_size * fov_size,
        MINIGRID_N_CELL_TYPES,
        theta_goal,
    )
    return _agent_step_result(action_distribution, horizon, beliefs)


def minigrid_sparse_active_step(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    observation_tensor: List[Float32],
    orientation_tensor: List[Float32],
    vision_image: List[Int],
    direction: Int,
    goal: List[Float32],
    action_prior: List[Float32],
    previous_action: Int,
    time_remaining: Int,
    planning_horizon: Int,
    n_inference_iterations: Int,
    n_planning_iterations: Int,
    damping: Float32,
    grid_size: Int,
    fov_size: Int,
    n_static: Int,
    theta_goal: Bool,
) -> List[Float32]:
    """One-hot observation convenience wrapper for sparse Active inference."""
    return minigrid_sparse_agent_step(
        PLANNER_ACTIVE_INFERENCE,
        q_current_state,
        q_static_state,
        transition_indices,
        observation_tensor,
        orientation_tensor,
        observation_to_onehot(vision_image, fov_size, fov_size),
        direction_to_onehot(direction),
        goal,
        action_prior,
        previous_action,
        time_remaining,
        planning_horizon,
        n_inference_iterations,
        n_planning_iterations,
        damping,
        grid_size,
        fov_size,
        n_static,
        theta_goal,
    )


def frozen_lake_loopy_bp_step(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    observation_tensor: List[Float32],
    binary_observation: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    previous_action: Int,
    time_remaining: Int,
    planning_horizon: Int,
    planning_iterations: Int,
    n_states: Int,
    n_static: Int,
    n_actions: Int,
    n_channels: Int,
) -> List[Float32]:
    """Compatibility wrapper for the original Frozen Lake Loopy-BP API."""
    return frozen_lake_agent_step(
        PLANNER_LOOPY_BP,
        q_current_state,
        q_static_state,
        transition_tensor,
        observation_tensor,
        binary_observation,
        goal_by_static,
        action_prior,
        previous_action,
        time_remaining,
        planning_horizon,
        planning_iterations,
        1.0,
        n_states,
        n_static,
        n_actions,
        n_channels,
        True,
    )
