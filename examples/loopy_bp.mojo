from std.collections import List

from aif_mojo.loopy_bp import loopy_bp_planning_sparse


def pair(a: Float32, b: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(a)
    result.append(b)
    return result^


def main():
    # Two states, two actions, and two possible transition configurations.
    # Shape: (old_state, action, theta). Theta 0 makes action 1 move to the
    # preferred state; theta 1 reverses the action meanings.
    var transitions = List[Int]()
    transitions.append(0)
    transitions.append(1)
    transitions.append(1)
    transitions.append(0)
    transitions.append(1)
    transitions.append(0)
    transitions.append(0)
    transitions.append(1)

    var action_distribution = loopy_bp_planning_sparse(
        pair(1.0, 0.0),  # current state belief
        pair(0.9, 0.1),  # theta belief
        transitions,
        pair(0.01, 0.99),  # terminal goal
        pair(0.5, 0.5),  # action prior
        1,  # horizon
        3,  # message-passing iterations
        2,  # states
        2,  # actions
        2,  # theta configurations
    )
    print(
        "action distribution:", action_distribution[0], action_distribution[1]
    )
