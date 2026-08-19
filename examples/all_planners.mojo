from std.collections import List

from aif_mojo.agent import N_PLANNER_KINDS, dispatch_planner_first_action


def pair(first: Float32, second: Float32) -> List[Float32]:
    var result = List[Float32]()
    result.append(first)
    result.append(second)
    return result^


def transition_tensor() -> List[Float32]:
    # T(new_state, old_state, theta, action), S=Theta=A=2.
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


def observation_tensor() -> List[Float32]:
    # B(channel, outcome, state, theta), F=1 and C=S=Theta=2.
    var result = List[Float32]()
    result.extend(pair(0.9, 0.2))
    result.extend(pair(0.3, 0.8))
    result.extend(pair(0.1, 0.8))
    result.extend(pair(0.7, 0.2))
    return result^


def theta_goal() -> List[Float32]:
    var result = List[Float32]()
    result.extend(pair(0.15, 0.75))
    result.extend(pair(0.85, 0.25))
    return result^


def planner_name(kind: Int) -> String:
    if kind == 0:
        return "loopy"
    if kind == 1:
        return "loopy-vbp"
    if kind == 2:
        return "region-extended"
    if kind == 3:
        return "dyn-channel"
    if kind == 4:
        return "nuijten"
    if kind == 5:
        return "vbp-channel"
    if kind == 6:
        return "precise-info-seeking"
    return "active-inference"


def main():
    for planner_kind in range(N_PLANNER_KINDS):
        var action = dispatch_planner_first_action(
            planner_kind,
            pair(0.65, 0.35),
            pair(0.55, 0.45),
            transition_tensor(),
            observation_tensor(),
            theta_goal(),
            pair(0.6, 0.4),
            List[Float32](),  # Empty selects dense Active Inference.
            List[Float32](),
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
        print(planner_name(planner_kind), action[0], action[1])
