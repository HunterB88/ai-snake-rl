import torch


def encode_state(env):
    """Convert the Snake environment into 11 neural network inputs."""

    head_x, head_y = env.snake[0]

    # Current direction
    direction = env.DIRECTIONS[env.direction_index]

    # Directions relative to snake
    straight = direction

    right = env.DIRECTIONS[
        (env.direction_index + 1) % 4
    ]

    left = env.DIRECTIONS[
        (env.direction_index - 1) % 4
    ]

    def is_danger(move_direction):
        """Check whether moving in a direction causes a collision."""

        dx, dy = move_direction

        next_head = (
            head_x + dx,
            head_y + dy,
        )

        # Check walls
        hit_wall = (
            next_head[0] < 0
            or next_head[0] >= env.width
            or next_head[1] < 0
            or next_head[1] >= env.height
        )

        # Check body, allowing the tail to move away
        eating = next_head == env.food
        occupied = env.snake if eating else env.snake[:-1]

        hit_body = next_head in occupied

        return int(hit_wall or hit_body)

    # Danger inputs
    danger_straight = is_danger(straight)
    danger_right = is_danger(right)
    danger_left = is_danger(left)

    # Direction inputs
    moving_up = int(direction == (0, -1))
    moving_down = int(direction == (0, 1))
    moving_left = int(direction == (-1, 0))
    moving_right = int(direction == (1, 0))

    # Food location inputs
    if env.food is not None:
        food_x, food_y = env.food

        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)

    else:
        food_up = 0
        food_down = 0
        food_left = 0
        food_right = 0

    # Combine all 11 features
    state = [
        danger_straight,
        danger_right,
        danger_left,
        moving_up,
        moving_down,
        moving_left,
        moving_right,
        food_up,
        food_down,
        food_left,
        food_right,
    ]

    return torch.tensor(state, dtype=torch.float32)