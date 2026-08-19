from std.collections import List
from std.algorithm import parallelize
from std.math import exp, log

from aif_mojo.numerics import logsumexp, safe_log, softmax
from aif_mojo.sparse_messages import sparse_dyn_to_theta, sparse_reduced


def _zeros(length: Int) -> List[Float32]:
    var result = List[Float32](capacity=length)
    for _ in range(length):
        result.append(0.0)
    return result^


def _safe_log_values(values: List[Float32]) -> List[Float32]:
    var result = List[Float32](capacity=len(values))
    for value in values:
        result.append(safe_log(value))
    return result^


def forward_pass(
    log_reduced_per_t: List[Float32],
    log_q_x0: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Propagate normalized log state messages forward through time."""
    debug_assert(
        len(log_reduced_per_t) == horizon * n_states * n_states * n_actions,
        "reduced transition shape mismatch",
    )
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[state_idx] = log_q_x0[state_idx]

    for time_idx in range(horizon):
        var next_message = List[Float32](capacity=n_states)
        for new_idx in range(n_states):
            var terms = List[Float32](capacity=n_states * n_actions)
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[time_idx * n_states + old_idx]
                        + log_action_prior[action_idx]
                    )
            next_message.append(logsumexp(terms))
        var normalizer = logsumexp(next_message)
        for state_idx in range(n_states):
            result[(time_idx + 1) * n_states + state_idx] = (
                next_message[state_idx] - normalizer
            )
    return result^


def backward_messages(
    log_reduced_per_t: List[Float32],
    log_goal: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Propagate normalized log goal messages backward through time."""
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[horizon * n_states + state_idx] = log_goal[state_idx]

    for reverse_idx in range(horizon):
        var time_idx = horizon - 1 - reverse_idx
        var message = List[Float32](capacity=n_states)
        for old_idx in range(n_states):
            var terms = List[Float32](capacity=n_states * n_actions)
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[(time_idx + 1) * n_states + new_idx]
                        + log_action_prior[action_idx]
                    )
            message.append(logsumexp(terms))
        var normalizer = logsumexp(message)
        for state_idx in range(n_states):
            result[time_idx * n_states + state_idx] = (
                message[state_idx] - normalizer
            )
    return result^


def _forward_pass_with_local(
    log_reduced_per_t: List[Float32],
    log_q_x0: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = _zeros((horizon + 1) * n_states)
    for state_idx in range(n_states):
        result[state_idx] = log_q_x0[state_idx]

    for time_idx in range(horizon):
        var next_message = List[Float32](capacity=n_states)
        for new_idx in range(n_states):
            var terms = List[Float32](capacity=n_states * n_actions)
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[time_idx * n_states + old_idx]
                        + log_local_to_x[time_idx * n_states + old_idx]
                        + log_action_prior[action_idx]
                    )
            next_message.append(logsumexp(terms))
        var normalizer = logsumexp(next_message)
        for state_idx in range(n_states):
            result[(time_idx + 1) * n_states + state_idx] = (
                next_message[state_idx] - normalizer
            )
    return result^


def _backward_messages_with_local(
    log_reduced_per_t: List[Float32],
    log_action_prior: List[Float32],
    log_local_to_x: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    var result = _zeros((horizon + 1) * n_states)
    for reverse_idx in range(horizon):
        var time_idx = horizon - 1 - reverse_idx
        var message = List[Float32](capacity=n_states)
        for old_idx in range(n_states):
            var terms = List[Float32](capacity=n_states * n_actions)
            for new_idx in range(n_states):
                for action_idx in range(n_actions):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + result[(time_idx + 1) * n_states + new_idx]
                        + log_local_to_x[(time_idx + 1) * n_states + new_idx]
                        + log_action_prior[action_idx]
                    )
            message.append(logsumexp(terms))
        var normalizer = logsumexp(message)
        for state_idx in range(n_states):
            result[time_idx * n_states + state_idx] = (
                message[state_idx] - normalizer
            )
    return result^


def _add_messages(lhs: List[Float32], rhs: List[Float32]) -> List[Float32]:
    debug_assert(len(lhs) == len(rhs), "message shape mismatch")
    var result = List[Float32](capacity=len(lhs))
    for i in range(len(lhs)):
        result.append(lhs[i] + rhs[i])
    return result^


def compute_action_marginals(
    log_reduced_per_t: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
) -> List[Float32]:
    """Compute probability-space action marginals for every time step."""
    var result = List[Float32](capacity=horizon * n_actions)
    for time_idx in range(horizon):
        var action_logits = List[Float32](capacity=n_actions)
        for action_idx in range(n_actions):
            var terms = List[Float32](capacity=n_states * n_states)
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    var reduced_offset = (
                        (time_idx * n_states + new_idx) * n_states + old_idx
                    ) * n_actions + action_idx
                    terms.append(
                        log_reduced_per_t[reduced_offset]
                        + log_bwd_messages[(time_idx + 1) * n_states + new_idx]
                        + log_fwd_messages[time_idx * n_states + old_idx]
                    )
            action_logits.append(
                logsumexp(terms) + log_action_prior[action_idx]
            )
        var probabilities = softmax(action_logits)
        for action_idx in range(n_actions):
            result.append(probabilities[action_idx])
    return result^


def compute_theta_cavities(
    log_prior_theta: List[Float32],
    log_dyn_to_theta: List[Float32],
    horizon: Int,
    n_static: Int,
) -> List[Float32]:
    """Normalize prior plus every dynamics message except the local factor."""
    debug_assert(
        len(log_dyn_to_theta) == horizon * n_static,
        "dynamics message shape mismatch",
    )
    var totals = List[Float32](capacity=n_static)
    for static_idx in range(n_static):
        var value = log_prior_theta[static_idx]
        for time_idx in range(horizon):
            value += log_dyn_to_theta[time_idx * n_static + static_idx]
        totals.append(value)

    var result = List[Float32](capacity=horizon * n_static)
    for time_idx in range(horizon):
        var cavity = List[Float32](capacity=n_static)
        for static_idx in range(n_static):
            cavity.append(
                totals[static_idx]
                - log_dyn_to_theta[time_idx * n_static + static_idx]
            )
        var normalizer = logsumexp(cavity)
        for static_idx in range(n_static):
            result.append(cavity[static_idx] - normalizer)
    return result^


def compute_reduced_per_t_dense(
    log_transition: List[Float32],
    log_cavity_theta: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Marginalize theta from dense (new, old, theta, action) transitions."""
    var result = List[Float32](
        capacity=horizon * n_states * n_states * n_actions
    )
    for time_idx in range(horizon):
        for new_idx in range(n_states):
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var terms = List[Float32](capacity=n_static)
                    for static_idx in range(n_static):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        terms.append(
                            log_transition[transition_offset]
                            + log_cavity_theta[time_idx * n_static + static_idx]
                        )
                    result.append(logsumexp(terms))
    return result^


def compute_dyn_to_theta_dense(
    log_transition: List[Float32],
    log_fwd_messages: List[Float32],
    log_bwd_messages: List[Float32],
    log_action_prior: List[Float32],
    horizon: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Compute dense dynamics-factor messages to theta."""
    var result = List[Float32](capacity=horizon * n_static)
    for time_idx in range(horizon):
        for static_idx in range(n_static):
            var terms = List[Float32](capacity=n_states * n_states * n_actions)
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    for action_idx in range(n_actions):
                        var transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        terms.append(
                            log_transition[transition_offset]
                            + log_fwd_messages[time_idx * n_states + old_idx]
                            + log_bwd_messages[
                                (time_idx + 1) * n_states + new_idx
                            ]
                            + log_action_prior[action_idx]
                        )
            result.append(logsumexp(terms))
    return result^


def dense_loopy_bp_workspace_size(
    horizon: Int, n_states: Int, n_actions: Int, n_static: Int
) -> Int:
    """Number of Float32 slots used by the allocation-free dense kernel."""
    return (
        2 * horizon * n_static
        + horizon * n_states * n_states * n_actions
        + 2 * (horizon + 1) * n_states
        + horizon * n_actions
        + n_static
    )


def make_dense_loopy_bp_workspace(
    horizon: Int, n_states: Int, n_actions: Int, n_static: Int
) -> List[Float32]:
    """Allocate one workspace that can be reused across dense planner calls."""
    return _zeros(
        dense_loopy_bp_workspace_size(horizon, n_states, n_actions, n_static)
    )


struct PreparedDenseLoopyBP:
    """Reusable terminal-goal planner with static logs and workspace cached."""

    var log_transition: List[Float32]
    var log_goal: List[Float32]
    var log_action_prior: List[Float32]
    var workspace: List[Float32]
    var horizon: Int
    var n_iterations: Int
    var n_states: Int
    var n_actions: Int
    var n_static: Int

    def __init__(
        out self,
        transition_tensor: List[Float32],
        goal: List[Float32],
        action_prior: List[Float32],
        horizon: Int,
        n_iterations: Int,
        n_states: Int,
        n_actions: Int,
        n_static: Int,
    ):
        self.log_transition = _safe_log_values(transition_tensor)
        self.log_goal = _safe_log_values(goal)
        self.log_action_prior = _safe_log_values(action_prior)
        self.workspace = make_dense_loopy_bp_workspace(
            horizon, n_states, n_actions, n_static
        )
        self.horizon = horizon
        self.n_iterations = n_iterations
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_static = n_static

    def plan(
        mut self,
        q_current_state: List[Float32],
        q_static_state: List[Float32],
    ) -> List[Float32]:
        """Plan while reusing cached static logs and mutable scratch storage."""
        var log_q0 = _safe_log_values(q_current_state)
        var log_prior_theta = _safe_log_values(q_static_state)
        return loopy_bp_planning_dense_prelogged_with_workspace(
            log_q0,
            log_prior_theta,
            self.log_transition,
            self.log_goal,
            self.log_action_prior,
            self.workspace,
            self.horizon,
            self.n_iterations,
            self.n_states,
            self.n_actions,
            self.n_static,
        )


def _normalize_log_workspace(
    mut workspace: List[Float32], start: Int, length: Int
):
    """Normalize one contiguous log-space workspace slice in place."""
    var maximum = workspace[start]
    for idx in range(1, length):
        maximum = max(maximum, workspace[start + idx])
    var shifted_sum = Float32(0.0)
    for idx in range(length):
        shifted_sum += exp(workspace[start + idx] - maximum)
    var normalizer = maximum + log(shifted_sum)
    for idx in range(length):
        workspace[start + idx] -= normalizer


def loopy_bp_planning_dense_prelogged_with_workspace(
    log_q0: List[Float32],
    log_prior_theta: List[Float32],
    log_transition: List[Float32],
    log_goal: List[Float32],
    log_action_prior: List[Float32],
    mut workspace: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense terminal-goal Loopy BP without hot-loop allocations.

    Inputs are already in log space. The caller-owned workspace is reusable,
    which removes both the per-reduction temporary Lists and the large
    per-iteration message allocations used by the compatibility helpers.
    """
    debug_assert(len(log_q0) == n_states, "state belief shape mismatch")
    debug_assert(
        len(log_prior_theta) == n_static, "theta belief shape mismatch"
    )
    debug_assert(
        len(log_transition) == n_states * n_states * n_static * n_actions,
        "transition shape mismatch",
    )
    debug_assert(len(log_goal) == n_states, "goal shape mismatch")
    debug_assert(
        len(log_action_prior) == n_actions, "action prior shape mismatch"
    )
    debug_assert(
        len(workspace)
        == dense_loopy_bp_workspace_size(
            horizon, n_states, n_actions, n_static
        ),
        "dense Loopy-BP workspace shape mismatch",
    )

    var cavity_start = 0
    var reduced_start = cavity_start + horizon * n_static
    var fwd_start = reduced_start + horizon * n_states * n_states * n_actions
    var bwd_start = fwd_start + (horizon + 1) * n_states
    var action_start = bwd_start + (horizon + 1) * n_states
    var dyn_start = action_start + horizon * n_actions
    var totals_start = dyn_start + horizon * n_static
    var workspace_ptr = workspace.unsafe_ptr()
    var action_prior_ptr = log_action_prior.unsafe_ptr()
    var transition_ptr = log_transition.unsafe_ptr()

    for time_idx in range(horizon):
        for static_idx in range(n_static):
            workspace[
                cavity_start + time_idx * n_static + static_idx
            ] = log_prior_theta[static_idx]

    for _ in range(n_iterations):
        # Marginalize theta directly into the persistent reduced buffer.
        @parameter
        def fill_reduced_row(row_idx: Int) capturing:
            var time_idx = row_idx // n_states
            var new_idx = row_idx % n_states
            for old_idx in range(n_states):
                for action_idx in range(n_actions):
                    var transition_offset = (
                        (new_idx * n_states + old_idx) * n_static
                    ) * n_actions + action_idx
                    var maximum = (
                        transition_ptr[transition_offset]
                        + workspace_ptr[cavity_start + time_idx * n_static]
                    )
                    for static_idx in range(1, n_static):
                        var term = (
                            transition_ptr[
                                transition_offset + static_idx * n_actions
                            ]
                            + workspace_ptr[
                                cavity_start + time_idx * n_static + static_idx
                            ]
                        )
                        maximum = max(maximum, term)
                    var shifted_sum = Float32(0.0)
                    for static_idx in range(n_static):
                        shifted_sum += exp(
                            transition_ptr[
                                transition_offset + static_idx * n_actions
                            ]
                            + workspace_ptr[
                                cavity_start + time_idx * n_static + static_idx
                            ]
                            - maximum
                        )
                    var reduced_offset = (
                        ((time_idx * n_states + new_idx) * n_states) + old_idx
                    ) * n_actions + action_idx
                    workspace_ptr[
                        reduced_start + reduced_offset
                    ] = maximum + log(shifted_sum)

        if n_states >= 32:
            parallelize[fill_reduced_row](horizon * n_states)
        else:
            for row_idx in range(horizon * n_states):
                fill_reduced_row(row_idx)

        # Forward messages. Each reduction streams twice over the source
        # slice instead of constructing a temporary terms List.
        for state_idx in range(n_states):
            workspace[fwd_start + state_idx] = log_q0[state_idx]
        for time_idx in range(horizon):
            var next_start = fwd_start + (time_idx + 1) * n_states
            for new_idx in range(n_states):
                var first_reduced_offset = (
                    (time_idx * n_states + new_idx) * n_states * n_actions
                )
                var maximum = (
                    workspace_ptr[reduced_start + first_reduced_offset]
                    + workspace_ptr[fwd_start + time_idx * n_states]
                    + action_prior_ptr[0]
                )
                for old_idx in range(n_states):
                    var action_idx = 0
                    while action_idx + 4 <= n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        var terms = (
                            workspace_ptr.load[width=4](
                                reduced_start + reduced_offset
                            )
                            + action_prior_ptr.load[width=4](action_idx)
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                        )
                        maximum = max(maximum, terms.reduce_max())
                        action_idx += 4
                    while action_idx < n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        maximum = max(
                            maximum,
                            workspace_ptr[reduced_start + reduced_offset]
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            + action_prior_ptr[action_idx],
                        )
                        action_idx += 1
                var shifted_sum = Float32(0.0)
                for old_idx in range(n_states):
                    var action_idx = 0
                    while action_idx + 4 <= n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        var terms = (
                            workspace_ptr.load[width=4](
                                reduced_start + reduced_offset
                            )
                            + action_prior_ptr.load[width=4](action_idx)
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            - maximum
                        )
                        shifted_sum += exp(terms).reduce_add()
                        action_idx += 4
                    while action_idx < n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        shifted_sum += exp(
                            workspace_ptr[reduced_start + reduced_offset]
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            + action_prior_ptr[action_idx]
                            - maximum
                        )
                        action_idx += 1
                workspace_ptr[next_start + new_idx] = maximum + log(shifted_sum)
            _normalize_log_workspace(workspace, next_start, n_states)

        # Backward messages.
        var terminal_start = bwd_start + horizon * n_states
        for state_idx in range(n_states):
            workspace[terminal_start + state_idx] = log_goal[state_idx]
        for reverse_idx in range(horizon):
            var time_idx = horizon - 1 - reverse_idx
            var message_start = bwd_start + time_idx * n_states
            for old_idx in range(n_states):
                var first_reduced_offset = (
                    time_idx * n_states * n_states + old_idx
                ) * n_actions
                var maximum = (
                    workspace_ptr[reduced_start + first_reduced_offset]
                    + workspace_ptr[bwd_start + (time_idx + 1) * n_states]
                    + action_prior_ptr[0]
                )
                for new_idx in range(n_states):
                    var action_idx = 0
                    while action_idx + 4 <= n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        var terms = (
                            workspace_ptr.load[width=4](
                                reduced_start + reduced_offset
                            )
                            + action_prior_ptr.load[width=4](action_idx)
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                        )
                        maximum = max(maximum, terms.reduce_max())
                        action_idx += 4
                    while action_idx < n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        maximum = max(
                            maximum,
                            workspace_ptr[reduced_start + reduced_offset]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            + action_prior_ptr[action_idx],
                        )
                        action_idx += 1
                var shifted_sum = Float32(0.0)
                for new_idx in range(n_states):
                    var action_idx = 0
                    while action_idx + 4 <= n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        var terms = (
                            workspace_ptr.load[width=4](
                                reduced_start + reduced_offset
                            )
                            + action_prior_ptr.load[width=4](action_idx)
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            - maximum
                        )
                        shifted_sum += exp(terms).reduce_add()
                        action_idx += 4
                    while action_idx < n_actions:
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        shifted_sum += exp(
                            workspace_ptr[reduced_start + reduced_offset]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            + action_prior_ptr[action_idx]
                            - maximum
                        )
                        action_idx += 1
                workspace_ptr[message_start + old_idx] = maximum + log(
                    shifted_sum
                )
            _normalize_log_workspace(workspace, message_start, n_states)

        # Action marginals, using the existing action slice for logits and
        # probabilities so no scratch allocation is required.
        for time_idx in range(horizon):
            var logits_start = action_start + time_idx * n_actions
            for action_idx in range(n_actions):
                var first_reduced_offset = (
                    time_idx * n_states * n_states * n_actions + action_idx
                )
                var maximum = (
                    workspace_ptr[reduced_start + first_reduced_offset]
                    + workspace_ptr[bwd_start + (time_idx + 1) * n_states]
                    + workspace_ptr[fwd_start + time_idx * n_states]
                )
                for new_idx in range(n_states):
                    for old_idx in range(n_states):
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        var term = (
                            workspace_ptr[reduced_start + reduced_offset]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                        )
                        maximum = max(maximum, term)
                var shifted_sum = Float32(0.0)
                for new_idx in range(n_states):
                    for old_idx in range(n_states):
                        var reduced_offset = (
                            ((time_idx * n_states + new_idx) * n_states)
                            + old_idx
                        ) * n_actions + action_idx
                        shifted_sum += exp(
                            workspace_ptr[reduced_start + reduced_offset]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            - maximum
                        )
                workspace_ptr[logits_start + action_idx] = (
                    maximum + log(shifted_sum) + action_prior_ptr[action_idx]
                )
            var maximum = workspace[logits_start]
            for action_idx in range(1, n_actions):
                maximum = max(maximum, workspace[logits_start + action_idx])
            var total = Float32(0.0)
            for action_idx in range(n_actions):
                var probability = exp(
                    workspace[logits_start + action_idx] - maximum
                )
                workspace[logits_start + action_idx] = probability
                total += probability
            for action_idx in range(n_actions):
                workspace[logits_start + action_idx] /= total

        # Dynamics-factor messages to theta.
        @parameter
        def fill_dyn_message(message_idx: Int) capturing:
            var time_idx = message_idx // n_static
            var static_idx = message_idx % n_static
            var transition_offset = static_idx * n_actions
            var maximum = (
                transition_ptr[transition_offset]
                + workspace_ptr[fwd_start + time_idx * n_states]
                + workspace_ptr[bwd_start + (time_idx + 1) * n_states]
                + action_prior_ptr[0]
            )
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    var action_idx = 0
                    while action_idx + 4 <= n_actions:
                        transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var terms = (
                            transition_ptr.load[width=4](transition_offset)
                            + action_prior_ptr.load[width=4](action_idx)
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                        )
                        maximum = max(maximum, terms.reduce_max())
                        action_idx += 4
                    while action_idx < n_actions:
                        transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        maximum = max(
                            maximum,
                            transition_ptr[transition_offset]
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            + action_prior_ptr[action_idx],
                        )
                        action_idx += 1
            var shifted_sum = Float32(0.0)
            for new_idx in range(n_states):
                for old_idx in range(n_states):
                    var action_idx = 0
                    while action_idx + 4 <= n_actions:
                        transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        var terms = (
                            transition_ptr.load[width=4](transition_offset)
                            + action_prior_ptr.load[width=4](action_idx)
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            - maximum
                        )
                        shifted_sum += exp(terms).reduce_add()
                        action_idx += 4
                    while action_idx < n_actions:
                        transition_offset = (
                            (new_idx * n_states + old_idx) * n_static
                            + static_idx
                        ) * n_actions + action_idx
                        shifted_sum += exp(
                            transition_ptr[transition_offset]
                            + workspace_ptr[
                                fwd_start + time_idx * n_states + old_idx
                            ]
                            + workspace_ptr[
                                bwd_start + (time_idx + 1) * n_states + new_idx
                            ]
                            + action_prior_ptr[action_idx]
                            - maximum
                        )
                        action_idx += 1
            workspace_ptr[dyn_start + message_idx] = maximum + log(shifted_sum)

        if n_states >= 32:
            parallelize[fill_dyn_message](horizon * n_static)
        else:
            for message_idx in range(horizon * n_static):
                fill_dyn_message(message_idx)

        # Theta cavities. Totals and cavities live in the same workspace.
        for static_idx in range(n_static):
            var value = log_prior_theta[static_idx]
            for time_idx in range(horizon):
                value += workspace[dyn_start + time_idx * n_static + static_idx]
            workspace[totals_start + static_idx] = value
        for time_idx in range(horizon):
            var current_cavity_start = cavity_start + time_idx * n_static
            for static_idx in range(n_static):
                workspace[current_cavity_start + static_idx] = (
                    workspace[totals_start + static_idx]
                    - workspace[dyn_start + time_idx * n_static + static_idx]
                )
            _normalize_log_workspace(workspace, current_cavity_start, n_static)

    var result = List[Float32](capacity=n_actions)
    for action_idx in range(n_actions):
        result.append(workspace[action_start + action_idx])
    return result^


def loopy_bp_planning_sparse(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Sparse loopy-BP planner for the original one-dimensional terminal goal.
    """
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal)
    var log_action_prior = _safe_log_values(action_prior)

    var log_cavity_theta = List[Float32](capacity=horizon * n_static)
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var local_messages = _zeros((horizon + 1) * n_states)
    var action_per_t = List[Float32]()
    for _ in range(horizon):
        for action_idx in range(n_actions):
            action_per_t.append(log_action_prior[action_idx])

    var q_u = _zeros(horizon * n_actions)
    for _ in range(n_iterations):
        var reduced = sparse_reduced(
            transition_indices,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        var log_fwd = forward_pass(
            reduced, log_q0, log_action_prior, horizon, n_states, n_actions
        )
        var log_bwd = backward_messages(
            reduced, log_goal, log_action_prior, horizon, n_states, n_actions
        )
        q_u = compute_action_marginals(
            reduced,
            log_fwd,
            log_bwd,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        var log_dyn_to_theta = sparse_dyn_to_theta(
            transition_indices,
            log_fwd,
            log_bwd,
            local_messages,
            action_per_t,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, log_dyn_to_theta, horizon, n_static
        )

    var result = List[Float32](capacity=n_actions)
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    return result^


def loopy_bp_planning_dense(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    goal: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense Loopy BP for the original one-dimensional terminal goal.

    This compatibility entry point accepts probability-space inputs. Repeated
    callers can avoid log conversion and workspace allocation with
    `loopy_bp_planning_dense_prelogged_with_workspace`.
    """
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal)
    var log_action_prior = _safe_log_values(action_prior)
    var log_transition = _safe_log_values(transition_tensor)
    var workspace = make_dense_loopy_bp_workspace(
        horizon, n_states, n_actions, n_static
    )
    return loopy_bp_planning_dense_prelogged_with_workspace(
        log_q0,
        log_prior_theta,
        log_transition,
        log_goal,
        log_action_prior,
        workspace,
        horizon,
        n_iterations,
        n_states,
        n_actions,
        n_static,
    )


def loopy_bp_planning_dense_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_tensor: List[Float32],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Dense Loopy BP with a (state, static) preference at every time step."""
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal_by_static)
    var log_action_prior = _safe_log_values(action_prior)
    var log_transition = _safe_log_values(transition_tensor)

    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var previous_dyn_to_theta = _zeros(horizon * n_static)
    var q_u = _zeros(horizon * n_actions)
    for _ in range(n_iterations):
        var reduced = compute_reduced_per_t_dense(
            log_transition,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )

        var theta_logits = List[Float32]()
        for static_idx in range(n_static):
            var value = log_prior_theta[static_idx]
            for time_idx in range(horizon):
                value += previous_dyn_to_theta[time_idx * n_static + static_idx]
            theta_logits.append(value)
        var theta_normalizer = logsumexp(theta_logits)
        for static_idx in range(n_static):
            theta_logits[static_idx] -= theta_normalizer

        var log_preference = List[Float32]()
        for state_idx in range(n_states):
            var terms = List[Float32]()
            for static_idx in range(n_static):
                terms.append(
                    log_goal[state_idx * n_static + static_idx]
                    + theta_logits[static_idx]
                )
            log_preference.append(logsumexp(terms))
        var preference_normalizer = logsumexp(log_preference)
        for state_idx in range(n_states):
            log_preference[state_idx] -= preference_normalizer

        var local_messages = List[Float32]()
        for _ in range(horizon + 1):
            for state_idx in range(n_states):
                local_messages.append(log_preference[state_idx])

        var log_fwd = _forward_pass_with_local(
            reduced,
            log_q0,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var log_bwd = _backward_messages_with_local(
            reduced,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var fwd_with_local = _add_messages(log_fwd, local_messages)
        var bwd_with_local = _add_messages(log_bwd, local_messages)
        q_u = compute_action_marginals(
            reduced,
            fwd_with_local,
            bwd_with_local,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        previous_dyn_to_theta = compute_dyn_to_theta_dense(
            log_transition,
            fwd_with_local,
            bwd_with_local,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, previous_dyn_to_theta, horizon, n_static
        )

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    return result^


def loopy_bp_planning_sparse_theta_goal(
    q_current_state: List[Float32],
    q_static_state: List[Float32],
    transition_indices: List[Int],
    goal_by_static: List[Float32],
    action_prior: List[Float32],
    horizon: Int,
    n_iterations: Int,
    n_states: Int,
    n_actions: Int,
    n_static: Int,
) -> List[Float32]:
    """Sparse Loopy BP with a (state, static) preference at every time step."""
    debug_assert(
        len(goal_by_static) == n_states * n_static,
        "theta-dependent goal shape mismatch",
    )
    var log_prior_theta = _safe_log_values(q_static_state)
    var log_q0 = _safe_log_values(q_current_state)
    var log_goal = _safe_log_values(goal_by_static)
    var log_action_prior = _safe_log_values(action_prior)

    var log_cavity_theta = List[Float32]()
    for _ in range(horizon):
        for static_idx in range(n_static):
            log_cavity_theta.append(log_prior_theta[static_idx])

    var action_per_t = List[Float32]()
    for _ in range(horizon):
        for action_idx in range(n_actions):
            action_per_t.append(log_action_prior[action_idx])

    var previous_dyn_to_theta = _zeros(horizon * n_static)
    var q_u = _zeros(horizon * n_actions)
    for _ in range(n_iterations):
        var reduced = sparse_reduced(
            transition_indices,
            log_cavity_theta,
            horizon,
            n_states,
            n_actions,
            n_static,
        )

        var theta_logits = List[Float32]()
        for static_idx in range(n_static):
            var value = log_prior_theta[static_idx]
            for time_idx in range(horizon):
                value += previous_dyn_to_theta[time_idx * n_static + static_idx]
            theta_logits.append(value)
        var log_q_theta_normalizer = logsumexp(theta_logits)
        for static_idx in range(n_static):
            theta_logits[static_idx] -= log_q_theta_normalizer

        var log_preference = List[Float32]()
        for state_idx in range(n_states):
            var terms = List[Float32]()
            for static_idx in range(n_static):
                terms.append(
                    log_goal[state_idx * n_static + static_idx]
                    + theta_logits[static_idx]
                )
            log_preference.append(logsumexp(terms))
        var preference_normalizer = logsumexp(log_preference)
        for state_idx in range(n_states):
            log_preference[state_idx] -= preference_normalizer

        var local_messages = List[Float32]()
        for _ in range(horizon + 1):
            for state_idx in range(n_states):
                local_messages.append(log_preference[state_idx])

        var log_fwd = _forward_pass_with_local(
            reduced,
            log_q0,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var log_bwd = _backward_messages_with_local(
            reduced,
            log_action_prior,
            local_messages,
            horizon,
            n_states,
            n_actions,
        )
        var fwd_with_local = _add_messages(log_fwd, local_messages)
        var bwd_with_local = _add_messages(log_bwd, local_messages)
        q_u = compute_action_marginals(
            reduced,
            fwd_with_local,
            bwd_with_local,
            log_action_prior,
            horizon,
            n_states,
            n_actions,
        )
        previous_dyn_to_theta = sparse_dyn_to_theta(
            transition_indices,
            log_fwd,
            log_bwd,
            local_messages,
            action_per_t,
            horizon,
            n_states,
            n_actions,
            n_static,
        )
        log_cavity_theta = compute_theta_cavities(
            log_prior_theta, previous_dyn_to_theta, horizon, n_static
        )

    var result = List[Float32]()
    for action_idx in range(n_actions):
        result.append(q_u[action_idx])
    return result^
