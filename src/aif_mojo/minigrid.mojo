from std.collections import List


# Int-compatible counterparts of the JAX IntEnum values.
comptime MINIGRID_ACTION_TURN_LEFT = 0
comptime MINIGRID_ACTION_TURN_RIGHT = 1
comptime MINIGRID_ACTION_FORWARD = 2
comptime MINIGRID_ACTION_PICKUP = 3
comptime MINIGRID_ACTION_DROP = 4
comptime MINIGRID_ACTION_TOGGLE = 5
comptime MINIGRID_ACTION_DONE = 6

comptime MINIGRID_CELL_UNSEEN = 0
comptime MINIGRID_CELL_EMPTY = 1
comptime MINIGRID_CELL_WALL = 2
comptime MINIGRID_CELL_FLOOR = 3
comptime MINIGRID_CELL_DOOR = 4
comptime MINIGRID_CELL_KEY = 5
comptime MINIGRID_CELL_BALL = 6
comptime MINIGRID_CELL_BOX = 7
comptime MINIGRID_CELL_GOAL = 8
comptime MINIGRID_CELL_LAVA = 9
comptime MINIGRID_CELL_AGENT = 10

comptime MINIGRID_ORIENTATION_RIGHT = 0
comptime MINIGRID_ORIENTATION_DOWN = 1
comptime MINIGRID_ORIENTATION_LEFT = 2
comptime MINIGRID_ORIENTATION_UP = 3

comptime MINIGRID_N_CELL_TYPES = 11
comptime MINIGRID_N_ORIENTATIONS = 4
comptime MINIGRID_N_ACTIONS = 7
comptime MINIGRID_N_DOOR_KEY_STATES = 3


def _zeros(length: Int) -> List[Float32]:
    var result = List[Float32]()
    for _ in range(length):
        result.append(0.0)
    return result^


def _filled_int(length: Int, value: Int) -> List[Int]:
    var result = List[Int]()
    for _ in range(length):
        result.append(value)
    return result^


def _abs_int(value: Int) -> Int:
    if value < 0:
        return -value
    return value


def _append_coordinate(mut coordinates: List[Int], x: Int, y: Int):
    coordinates.append(x)
    coordinates.append(y)


def _contains_coordinate(coordinates: List[Int], x: Int, y: Int) -> Bool:
    for idx in range(len(coordinates) // 2):
        if coordinates[2 * idx] == x and coordinates[2 * idx + 1] == y:
            return True
    return False


def state_to_coords(state: Int, grid_size: Int) -> List[Int]:
    var result = List[Int]()
    result.append(state // grid_size)
    result.append(state % grid_size)
    return result^


def coords_to_state(x: Int, y: Int, grid_size: Int) -> Int:
    return x * grid_size + y


def flatten_state_index(
    state: Int,
    orientation: Int,
    door_key_state: Int,
    n_states: Int,
    n_orientations: Int,
    n_door_key_states: Int,
) -> Int:
    debug_assert(state >= 0 and state < n_states, "state out of range")
    return (
        state * (n_orientations * n_door_key_states)
        + orientation * n_door_key_states
        + door_key_state
    )


def unflatten_state_index(
    flat_idx: Int,
    n_states: Int,
    n_orientations: Int,
    n_door_key_states: Int,
) -> List[Int]:
    debug_assert(
        flat_idx >= 0
        and flat_idx < n_states * n_orientations * n_door_key_states,
        "flat state out of range",
    )
    var result = List[Int]()
    var door_key_state = flat_idx % n_door_key_states
    var remainder = flat_idx // n_door_key_states
    var orientation = remainder % n_orientations
    result.append(remainder // n_orientations)
    result.append(orientation)
    result.append(door_key_state)
    return result^


def key_position(key_pos: Int, grid_size: Int) -> List[Int]:
    return state_to_coords(key_pos, grid_size)


def door_position(door_pos: Int, grid_size: Int) -> List[Int]:
    var result = List[Int]()
    result.append(door_pos // grid_size + 1)
    result.append(door_pos % grid_size)
    return result^


def get_valid_static_configs(grid_size: Int) -> List[Int]:
    """Return flattened (key_pos, door_pos) pairs with key_x < door_x."""
    var n_positions = grid_size * grid_size - 2 * grid_size
    var result = List[Int]()
    for key_pos in range(n_positions):
        var key_x = key_pos // grid_size
        for door_pos in range(n_positions):
            var door_x = door_pos // grid_size + 1
            if key_x < door_x:
                result.append(key_pos)
                result.append(door_pos)
    return result^


def get_relative_coords(
    agent_x: Int,
    agent_y: Int,
    orientation: Int,
    target_x: Int,
    target_y: Int,
) -> List[Int]:
    var dx = target_x - agent_x
    var dy = target_y - agent_y
    var result = List[Int]()
    if orientation == MINIGRID_ORIENTATION_RIGHT:
        result.append(-dy)
        result.append(dx)
    elif orientation == MINIGRID_ORIENTATION_DOWN:
        result.append(dx)
        result.append(dy)
    elif orientation == MINIGRID_ORIENTATION_LEFT:
        result.append(dy)
        result.append(-dx)
    else:
        result.append(-dx)
        result.append(-dy)
    return result^


def in_fov(rel_x: Int, rel_y: Int, fov_size: Int = 7) -> Bool:
    var half = fov_size // 2
    return rel_x >= -half and rel_x <= half and rel_y >= 0 and rel_y < fov_size


def relative_to_fov_coords(
    rel_x: Int, rel_y: Int, fov_size: Int = 7
) -> List[Int]:
    var result = List[Int]()
    result.append(fov_size // 2 - rel_x)
    result.append(fov_size - 1 - rel_y)
    return result^


def relative_to_absolute_coords(
    agent_x: Int,
    agent_y: Int,
    orientation: Int,
    rel_x: Int,
    rel_y: Int,
) -> List[Int]:
    var result = List[Int]()
    if orientation == MINIGRID_ORIENTATION_RIGHT:
        result.append(agent_x + rel_y)
        result.append(agent_y - rel_x)
    elif orientation == MINIGRID_ORIENTATION_DOWN:
        result.append(agent_x + rel_x)
        result.append(agent_y + rel_y)
    elif orientation == MINIGRID_ORIENTATION_LEFT:
        result.append(agent_x - rel_y)
        result.append(agent_y + rel_x)
    else:
        result.append(agent_x - rel_x)
        result.append(agent_y - rel_y)
    return result^


def create_wall_set(door_x: Int, door_y: Int, grid_size: Int) -> List[Int]:
    """Return flattened absolute wall-coordinate pairs."""
    var result = List[Int]()
    for y in range(grid_size):
        if y != door_y:
            _append_coordinate(result, door_x, y)
    for x in range(grid_size):
        _append_coordinate(result, x, -1)
        _append_coordinate(result, x, grid_size)
    for y in range(grid_size):
        _append_coordinate(result, -1, y)
        _append_coordinate(result, grid_size, y)
    _append_coordinate(result, -1, -1)
    _append_coordinate(result, grid_size, -1)
    _append_coordinate(result, -1, grid_size)
    _append_coordinate(result, grid_size, grid_size)
    return result^


def generate_visibility_mask(
    agent_x: Int,
    agent_y: Int,
    width: Int,
    height: Int,
    walls: List[Int],
) -> List[Int]:
    """Mirror MiniGrid's two directional visibility sweeps."""
    var mask = _filled_int(width * height, 0)
    mask[agent_x * height + agent_y] = 1
    for reverse_j in range(height):
        var j = height - 1 - reverse_j
        for i in range(width - 1):
            if mask[i * height + j] == 0:
                continue
            if _contains_coordinate(walls, i, j):
                continue
            mask[(i + 1) * height + j] = 1
            if j > 0:
                mask[(i + 1) * height + j - 1] = 1
                mask[i * height + j - 1] = 1

        for reverse_i in range(width - 1):
            var i = width - 1 - reverse_i
            if mask[i * height + j] == 0:
                continue
            if _contains_coordinate(walls, i, j):
                continue
            mask[(i - 1) * height + j] = 1
            if j > 0:
                mask[(i - 1) * height + j - 1] = 1
                mask[i * height + j - 1] = 1
    return mask^


def get_fov(
    agent_x: Int,
    agent_y: Int,
    orientation: Int,
    key_x: Int,
    key_y: Int,
    door_x: Int,
    door_y: Int,
    door_key_state: Int,
    grid_size: Int,
    fov_size: Int = 7,
) -> List[Int]:
    debug_assert(
        fov_size >= 3 and fov_size % 2 == 1,
        "FOV size must be odd and at least three",
    )
    var half = fov_size // 2
    var fov = _filled_int(fov_size * fov_size, MINIGRID_CELL_EMPTY)
    var walls = create_wall_set(door_x, door_y, grid_size)

    for wall_idx in range(len(walls) // 2):
        var relative = get_relative_coords(
            agent_x,
            agent_y,
            orientation,
            walls[2 * wall_idx],
            walls[2 * wall_idx + 1],
        )
        if in_fov(relative[0], relative[1], fov_size):
            var fov_coords = relative_to_fov_coords(
                relative[0], relative[1], fov_size
            )
            fov[fov_coords[0] * fov_size + fov_coords[1]] = MINIGRID_CELL_WALL

    var goal_relative = get_relative_coords(
        agent_x,
        agent_y,
        orientation,
        grid_size - 1,
        grid_size - 1,
    )
    if in_fov(goal_relative[0], goal_relative[1], fov_size):
        var goal_fov = relative_to_fov_coords(
            goal_relative[0], goal_relative[1], fov_size
        )
        fov[goal_fov[0] * fov_size + goal_fov[1]] = MINIGRID_CELL_GOAL

    if door_key_state == 0:
        var key_relative = get_relative_coords(
            agent_x, agent_y, orientation, key_x, key_y
        )
        if in_fov(key_relative[0], key_relative[1], fov_size):
            var key_fov = relative_to_fov_coords(
                key_relative[0], key_relative[1], fov_size
            )
            fov[key_fov[0] * fov_size + key_fov[1]] = MINIGRID_CELL_KEY

    var door_relative = get_relative_coords(
        agent_x, agent_y, orientation, door_x, door_y
    )
    if in_fov(door_relative[0], door_relative[1], fov_size):
        var door_fov = relative_to_fov_coords(
            door_relative[0], door_relative[1], fov_size
        )
        fov[door_fov[0] * fov_size + door_fov[1]] = MINIGRID_CELL_DOOR

    if door_key_state >= 1:
        fov[half * fov_size + fov_size - 1] = MINIGRID_CELL_KEY
    if door_key_state != 2:
        _append_coordinate(walls, door_x, door_y)

    var relative_walls = List[Int]()
    for wall_idx in range(len(walls) // 2):
        var relative = get_relative_coords(
            agent_x,
            agent_y,
            orientation,
            walls[2 * wall_idx],
            walls[2 * wall_idx + 1],
        )
        if in_fov(relative[0], relative[1], fov_size):
            var coords = relative_to_fov_coords(
                relative[0], relative[1], fov_size
            )
            _append_coordinate(relative_walls, coords[0], coords[1])

    var visibility = generate_visibility_mask(
        half,
        fov_size - 1,
        fov_size,
        fov_size,
        relative_walls,
    )
    for x_offset in range(2 * half + 1):
        var rel_x = x_offset - half
        for rel_y in range(fov_size):
            var coords = relative_to_fov_coords(rel_x, rel_y, fov_size)
            var offset = coords[0] * fov_size + coords[1]
            if visibility[offset] == 0:
                fov[offset] = MINIGRID_CELL_UNSEEN
    return fov^


def get_next_orientation(orientation: Int, action: Int) -> Int:
    if action == MINIGRID_ACTION_TURN_LEFT:
        return (orientation + 3) % MINIGRID_N_ORIENTATIONS
    if action == MINIGRID_ACTION_TURN_RIGHT:
        return (orientation + 1) % MINIGRID_N_ORIENTATIONS
    return orientation


def get_next_door_key_state(
    agent_x: Int,
    agent_y: Int,
    orientation: Int,
    key_x: Int,
    key_y: Int,
    door_x: Int,
    door_y: Int,
    action: Int,
    door_key_state: Int,
) -> Int:
    if action == MINIGRID_ACTION_PICKUP:
        if door_key_state > 0:
            return door_key_state
        var relative = get_relative_coords(
            agent_x, agent_y, orientation, key_x, key_y
        )
        if relative[0] == 0 and relative[1] == 1:
            return 1
        return door_key_state
    if action != MINIGRID_ACTION_TOGGLE or door_key_state != 1:
        return door_key_state
    var relative = get_relative_coords(
        agent_x, agent_y, orientation, door_x, door_y
    )
    if relative[0] == 0 and relative[1] == 1:
        return 2
    return door_key_state


def get_next_agent_position(
    agent_x: Int,
    agent_y: Int,
    orientation: Int,
    door_x: Int,
    door_y: Int,
    key_x: Int,
    key_y: Int,
    door_key_state: Int,
    action: Int,
    grid_size: Int,
) -> Int:
    if action != MINIGRID_ACTION_FORWARD:
        return coords_to_state(agent_x, agent_y, grid_size)
    var new_x = agent_x
    var new_y = agent_y
    if orientation == MINIGRID_ORIENTATION_RIGHT:
        new_x += 1
    elif orientation == MINIGRID_ORIENTATION_DOWN:
        new_y += 1
    elif orientation == MINIGRID_ORIENTATION_LEFT:
        new_x -= 1
    else:
        new_y -= 1
    if new_x < 0 or new_x >= grid_size or new_y < 0 or new_y >= grid_size:
        return coords_to_state(agent_x, agent_y, grid_size)
    if new_x == key_x and new_y == key_y and door_key_state == 0:
        return coords_to_state(agent_x, agent_y, grid_size)
    if new_x == door_x and door_key_state != 2:
        return coords_to_state(agent_x, agent_y, grid_size)
    if new_x == door_x and new_y != door_y:
        return coords_to_state(agent_x, agent_y, grid_size)
    return coords_to_state(new_x, new_y, grid_size)


def next_minigrid_state(
    old_state: Int,
    action: Int,
    grid_size: Int,
    key_pos: Int,
    door_pos: Int,
) -> Int:
    """Apply one deterministic action for an explicit static configuration."""
    var n_locations = grid_size * grid_size
    var unpacked = unflatten_state_index(
        old_state,
        n_locations,
        MINIGRID_N_ORIENTATIONS,
        MINIGRID_N_DOOR_KEY_STATES,
    )
    var location = unpacked[0]
    var orientation = unpacked[1]
    var door_key_state = unpacked[2]
    var agent_x = location // grid_size
    var agent_y = location % grid_size
    var key_x = key_pos // grid_size
    var key_y = key_pos % grid_size
    var door_x = door_pos // grid_size + 1
    var door_y = door_pos % grid_size
    # The JAX reference keeps otherwise-invalid wall states fully absorbing.
    if agent_x == door_x and agent_y != door_y:
        return old_state
    var new_location = get_next_agent_position(
        agent_x,
        agent_y,
        orientation,
        door_x,
        door_y,
        key_x,
        key_y,
        door_key_state,
        action,
        grid_size,
    )
    var new_door_key_state = get_next_door_key_state(
        agent_x,
        agent_y,
        orientation,
        key_x,
        key_y,
        door_x,
        door_y,
        action,
        door_key_state,
    )
    return flatten_state_index(
        new_location,
        get_next_orientation(orientation, action),
        new_door_key_state,
        n_locations,
        MINIGRID_N_ORIENTATIONS,
        MINIGRID_N_DOOR_KEY_STATES,
    )


def generate_minigrid_transition_indices(
    grid_size: Int, valid_configs: List[Int]
) -> List[Int]:
    """Return deterministic T_idx in (old, action, theta) order."""
    debug_assert(len(valid_configs) % 2 == 0, "static config shape mismatch")
    var n_static = len(valid_configs) // 2
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    var result = List[Int]()
    for old_state in range(n_states):
        for action in range(MINIGRID_N_ACTIONS):
            for theta in range(n_static):
                result.append(
                    next_minigrid_state(
                        old_state,
                        action,
                        grid_size,
                        valid_configs[2 * theta],
                        valid_configs[2 * theta + 1],
                    )
                )
    return result^


def generate_minigrid_transition_tensor(
    grid_size: Int, valid_configs: List[Int]
) -> List[Float32]:
    """Expand deterministic indices to dense T(new, old, theta, action).

    Dense MiniGrid tensors scale quadratically in state count. This helper is
    deliberately restricted to tiny correctness fixtures; real planning uses
    generate_minigrid_transition_indices.
    """
    debug_assert(grid_size <= 3, "dense MiniGrid generation is for tiny cases")
    var n_static = len(valid_configs) // 2
    var n_states = (
        grid_size
        * grid_size
        * MINIGRID_N_ORIENTATIONS
        * MINIGRID_N_DOOR_KEY_STATES
    )
    var indices = generate_minigrid_transition_indices(grid_size, valid_configs)
    var result = _zeros(n_states * n_states * n_static * MINIGRID_N_ACTIONS)
    for old_state in range(n_states):
        for action in range(MINIGRID_N_ACTIONS):
            for theta in range(n_static):
                var index_offset = (
                    old_state * MINIGRID_N_ACTIONS + action
                ) * n_static + theta
                var new_state = indices[index_offset]
                var dense_offset = (
                    (new_state * n_states + old_state) * n_static + theta
                ) * MINIGRID_N_ACTIONS + action
                result[dense_offset] = 1.0
    return result^


def generate_minigrid_observation_tensor(
    grid_size: Int, valid_configs: List[Int], fov_size: Int = 7
) -> List[Float32]:
    """Return hard B(fov_x, fov_y, cell_type, state, theta)."""
    debug_assert(len(valid_configs) % 2 == 0, "static config shape mismatch")
    var n_locations = grid_size * grid_size
    var n_states = (
        n_locations * MINIGRID_N_ORIENTATIONS * MINIGRID_N_DOOR_KEY_STATES
    )
    var n_static = len(valid_configs) // 2
    var result = _zeros(
        fov_size * fov_size * MINIGRID_N_CELL_TYPES * n_states * n_static
    )
    for location in range(n_locations):
        var agent_x = location // grid_size
        var agent_y = location % grid_size
        for orientation in range(MINIGRID_N_ORIENTATIONS):
            for theta in range(n_static):
                var key_pos = valid_configs[2 * theta]
                var door_pos = valid_configs[2 * theta + 1]
                var key_x = key_pos // grid_size
                var key_y = key_pos % grid_size
                var door_x = door_pos // grid_size + 1
                var door_y = door_pos % grid_size
                for door_key_state in range(MINIGRID_N_DOOR_KEY_STATES):
                    var fov = get_fov(
                        agent_x,
                        agent_y,
                        orientation,
                        key_x,
                        key_y,
                        door_x,
                        door_y,
                        door_key_state,
                        grid_size,
                        fov_size,
                    )
                    var state = flatten_state_index(
                        location,
                        orientation,
                        door_key_state,
                        n_locations,
                        MINIGRID_N_ORIENTATIONS,
                        MINIGRID_N_DOOR_KEY_STATES,
                    )
                    for fov_x in range(fov_size):
                        for fov_y in range(fov_size):
                            var cell = fov[fov_x * fov_size + fov_y]
                            var offset = (
                                (
                                    (fov_x * fov_size + fov_y)
                                    * MINIGRID_N_CELL_TYPES
                                    + cell
                                )
                                * n_states
                                + state
                            ) * n_static + theta
                            result[offset] = 1.0
    return result^


def soften_minigrid_observation_tensor(
    hard: List[Float32],
    fov_size: Int,
    alpha: Float32,
    n_states: Int,
    n_static: Int,
) -> List[Float32]:
    """Apply the reference Manhattan-distance precision to hard observations."""
    debug_assert(
        len(hard)
        == fov_size * fov_size * MINIGRID_N_CELL_TYPES * n_states * n_static,
        "observation shape mismatch",
    )
    var result = _zeros(len(hard))
    var half = fov_size // 2
    var reference_y = fov_size - 2
    var agent_y = fov_size - 1
    var uniform = 1.0 / Float32(MINIGRID_N_CELL_TYPES)
    for fov_x in range(fov_size):
        for fov_y in range(fov_size):
            var distance_to_reference = _abs_int(fov_x - half) + _abs_int(
                fov_y - reference_y
            )
            var distance_to_agent = _abs_int(fov_x - half) + _abs_int(
                fov_y - agent_y
            )
            var distance = min(distance_to_reference, distance_to_agent)
            var precision = 1.0 - alpha * Float32(distance)
            if precision < 0.0:
                precision = 0.0
            for cell in range(MINIGRID_N_CELL_TYPES):
                for state in range(n_states):
                    for theta in range(n_static):
                        var offset = (
                            (
                                (fov_x * fov_size + fov_y)
                                * MINIGRID_N_CELL_TYPES
                                + cell
                            )
                            * n_states
                            + state
                        ) * n_static + theta
                        result[offset] = (
                            precision * hard[offset]
                            + (1.0 - precision) * uniform
                        )
    return result^


def generate_minigrid_orientation_observation_tensor(
    grid_size: Int,
) -> List[Float32]:
    """Return B_orientation(orientation, state)."""
    var n_locations = grid_size * grid_size
    var n_states = (
        n_locations * MINIGRID_N_ORIENTATIONS * MINIGRID_N_DOOR_KEY_STATES
    )
    var result = _zeros(MINIGRID_N_ORIENTATIONS * n_states)
    for state in range(n_states):
        var orientation = (
            state // MINIGRID_N_DOOR_KEY_STATES
        ) % MINIGRID_N_ORIENTATIONS
        result[orientation * n_states + state] = 1.0
    return result^


def observation_to_onehot(
    image: List[Int], fov_width: Int, fov_height: Int
) -> List[Float32]:
    debug_assert(len(image) == fov_width * fov_height, "image shape mismatch")
    var result = _zeros(fov_width * fov_height * MINIGRID_N_CELL_TYPES)
    for idx in range(len(image)):
        debug_assert(
            image[idx] >= 0 and image[idx] < MINIGRID_N_CELL_TYPES,
            "cell type out of range",
        )
        result[idx * MINIGRID_N_CELL_TYPES + image[idx]] = 1.0
    return result^


def direction_to_onehot(direction: Int) -> List[Float32]:
    debug_assert(
        direction >= 0 and direction < MINIGRID_N_ORIENTATIONS,
        "direction out of range",
    )
    var result = _zeros(MINIGRID_N_ORIENTATIONS)
    result[direction] = 1.0
    return result^


def action_to_onehot(action: Int) -> List[Float32]:
    debug_assert(
        action >= 0 and action < MINIGRID_N_ACTIONS,
        "action out of range",
    )
    var result = _zeros(MINIGRID_N_ACTIONS)
    result[action] = 1.0
    return result^


def convert_action(action: Int) -> Int:
    return action


def _contains_cell(image: List[Int], cell: Int) -> Bool:
    for value in image:
        if value == cell:
            return True
    return False


def contains_key(image: List[Int]) -> Bool:
    return _contains_cell(image, MINIGRID_CELL_KEY)


def contains_door(image: List[Int]) -> Bool:
    return _contains_cell(image, MINIGRID_CELL_DOOR)


def generate_minigrid_goal(
    grid_size: Int, goal_x: Int, goal_y: Int
) -> List[Float32]:
    """Return a normalized goal over all orientations with the door open."""
    debug_assert(
        goal_x >= 0
        and goal_x < grid_size
        and goal_y >= 0
        and goal_y < grid_size,
        "goal coordinate out of range",
    )
    var n_locations = grid_size * grid_size
    var n_states = (
        n_locations * MINIGRID_N_ORIENTATIONS * MINIGRID_N_DOOR_KEY_STATES
    )
    var result = _zeros(n_states)
    var goal_location = coords_to_state(goal_x, goal_y, grid_size)
    for orientation in range(MINIGRID_N_ORIENTATIONS):
        var state = flatten_state_index(
            goal_location,
            orientation,
            2,
            n_locations,
            MINIGRID_N_ORIENTATIONS,
            MINIGRID_N_DOOR_KEY_STATES,
        )
        result[state] = 1.0 / Float32(MINIGRID_N_ORIENTATIONS)
    return result^
