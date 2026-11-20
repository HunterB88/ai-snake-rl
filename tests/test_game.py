from snake_env import SnakeEnv


def test_food_does_not_spawn_on_snake():
    env = SnakeEnv(seed=42)

    env.snake = [
        (x, y)
        for y in range(env.height)
        for x in range(env.width)
        if (x, y) != (0, 0)
    ]

    assert env._spawn_food() == (0, 0)


def test_no_food_when_board_is_full():
    env = SnakeEnv(seed=42)

    env.snake = [
        (x, y)
        for y in range(env.height)
        for x in range(env.width)
    ]

    assert env._spawn_food() is None