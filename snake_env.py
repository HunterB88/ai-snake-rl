import random


class SnakeEnv:
    """Snake game environment for reinforcement learning."""

    # Directions arranged clockwise
    DIRECTIONS = [
        (1, 0),   # Right
        (0, 1),   # Down
        (-1, 0),  # Left
        (0, -1),  # Up
    ]

    def __init__(self, width=20, height=20, seed=None):
        if width < 4 or height < 4:
            raise ValueError("Board must be at least 4x4")

        self.width = width
        self.height = height

        # Separate random number generator for reproducibility
        self.rng = random.Random(seed)

        self.reset()

    def reset(self):
        """Start a new game."""

        center_x = self.width // 2
        center_y = self.height // 2

        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]

        self.direction_index = 0

        self.score = 0
        self.done = False

        self.food = self._spawn_food()

        return self.get_state()

    def _spawn_food(self):
        """Place food in a random empty cell."""

        occupied = set(self.snake)

        available = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]

        if not available:
            return None

        return self.rng.choice(available)

    def get_state(self):
        """Return the current game state."""

        return {
            "snake": tuple(self.snake),
            "food": self.food,
            "direction": self.DIRECTIONS[self.direction_index],
            "score": self.score,
        }

    def step(self, action):
        """
        Perform one action.

        Actions:
            0 = Straight
            1 = Turn right
            2 = Turn left

        Returns:
            state, reward, done
        """

        if self.done:
            raise RuntimeError("Game is over. Call reset().")

        if action not in (0, 1, 2):
            raise ValueError("Action must be 0, 1, or 2")

        # Determine new direction
        if action == 1:
            self.direction_index = (self.direction_index + 1) % 4

        elif action == 2:
            self.direction_index = (self.direction_index - 1) % 4

        dx, dy = self.DIRECTIONS[self.direction_index]

        head_x, head_y = self.snake[0]

        new_head = (
            head_x + dx,
            head_y + dy,
        )

        # Check wall collision
        hit_wall = (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
        )

        eating = new_head == self.food

        # The tail moves away unless the snake eats
        occupied = self.snake if eating else self.snake[:-1]

        hit_body = new_head in occupied

        # Negative reward for dying
        if hit_wall or hit_body:
            self.done = True

            return self.get_state(), -10.0, True

        # Move snake
        self.snake.insert(0, new_head)

        if eating:
            self.score += 1

            # Positive reward for eating
            reward = 10.0

            self.food = self._spawn_food()

            # Game won if board is full
            if self.food is None:
                self.done = True

        else:
            self.snake.pop()

            # Small penalty per step
            reward = -0.01

        return self.get_state(), reward, self.done