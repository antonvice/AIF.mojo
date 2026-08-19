from std.benchmark import keep, run
from std.collections import List
from std.sys import argv

from aif_mojo.loopy_bp import loopy_bp_planning_dense


comptime N_STATIC = 2
comptime N_ACTIONS = 4
comptime HORIZON = 3
comptime PLANNING_ITERATIONS = 3
comptime WARMUP_ITERATIONS = 3


def state_belief(n_states: Int) -> List[Float32]:
    var result = List[Float32]()
    result.append(1.0)
    for _ in range(1, n_states):
        result.append(0.0)
    return result^


def static_belief() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.6)
    result.append(0.4)
    return result^


def action_prior() -> List[Float32]:
    var result = List[Float32]()
    result.append(0.4)
    result.append(0.3)
    result.append(0.2)
    result.append(0.1)
    return result^


def terminal_goal(n_states: Int) -> List[Float32]:
    var result = List[Float32]()
    var total = Float32(n_states * (n_states + 1) // 2)
    for state_idx in range(n_states):
        result.append(Float32(state_idx + 1) / total)
    return result^


def transition_tensor(n_states: Int) -> List[Float32]:
    """Deterministic T(new, old, theta, action) shared with JAX."""
    var result = List[Float32]()
    for new_state in range(n_states):
        for old_state in range(n_states):
            for theta in range(N_STATIC):
                for action in range(N_ACTIONS):
                    if new_state == (old_state + theta + action) % n_states:
                        result.append(1.0)
                    else:
                        result.append(0.0)
    return result^


def benchmark_fixture(name: String, n_states: Int) raises:
    var q_state = state_belief(n_states)
    var q_static = static_belief()
    var transition = transition_tensor(n_states)
    var goal = terminal_goal(n_states)
    var prior = action_prior()

    @parameter
    def plan_once() capturing:
        var policy = loopy_bp_planning_dense(
            q_state,
            q_static,
            transition,
            goal,
            prior,
            HORIZON,
            PLANNING_ITERATIONS,
            n_states,
            N_ACTIONS,
            N_STATIC,
        )
        keep(policy[0])

    var policy = loopy_bp_planning_dense(
        q_state,
        q_static,
        transition,
        goal,
        prior,
        HORIZON,
        PLANNING_ITERATIONS,
        n_states,
        N_ACTIONS,
        N_STATIC,
    )
    var report = run[func4=plan_once](
        num_warmup_iters=WARMUP_ITERATIONS,
        min_runtime_secs=0.25,
        max_runtime_secs=0.75,
        max_batch_size=1,
    )
    print(
        "RESULT",
        name,
        n_states,
        report.mean(),
        report.min(),
        report.max(),
        report.duration(),
        report.iters(),
        report.warmup_duration,
        policy[0],
        policy[1],
        policy[2],
        policy[3],
    )


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: fair_planner_benchmark <small|large>")
    if args[1] == "small":
        benchmark_fixture("small", 8)
        return
    if args[1] == "large":
        benchmark_fixture("large", 64)
        return
    raise Error("fixture must be small or large")
