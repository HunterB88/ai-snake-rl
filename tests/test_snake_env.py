import pytest

from snake_env import SnakeEnv


def test_reset():
    env = SnakeEnv(seed=42)

    assert env.score == 0
    assert env.done is False
    assert len(env.snake) == 3
    assert env.food not in env.snake


def test_move_straight():
    env = SnakeEnv(seed=42)

    old_head = env.snake[0]

    state, reward, done = env.step(0)

    assert env.snake[0] == (old_head[0] + 1, old_head[1])
    assert done is False


def test_turn_left():
    env = SnakeEnv(seed=42)

    old_head = env.snake[0]

    env.step(2)

    assert env.snake[0] == (old_head[0], old_head[1] - 1)


def test_eating_food():
    env = SnakeEnv(seed=42)

    head_x, head_y = env.snake[0]

    # Force food directly in front of snake
    env.food = (head_x + 1, head_y)

    old_length = len(env.snake)

    state, reward, done = env.step(0)

    assert env.score == 1
    assert len(env.snake) == old_length + 1
    assert reward == 10.0


def test_wall_collision():
    env = SnakeEnv(width=10, height=10, seed=42)

    while not env.done:
        state, reward, done = env.step(0)

    assert done is True
    assert reward == -10.0


def test_invalid_action():
    env = SnakeEnv(seed=42)

    with pytest.raises(ValueError):
        env.step(99)
