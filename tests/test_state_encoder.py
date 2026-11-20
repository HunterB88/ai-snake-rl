import torch

from snake_env import SnakeEnv
from state_encoder import encode_state


def test_state_has_11_features():
    env = SnakeEnv(seed=42)

    state = encode_state(env)

    assert state.shape == (11,)
    assert state.dtype == torch.float32


def test_state_contains_binary_values():
    env = SnakeEnv(seed=42)

    state = encode_state(env)

    assert all(value in (0, 1) for value in state.tolist())


def test_initial_direction_is_right():
    env = SnakeEnv(seed=42)

    state = encode_state(env)

    # Direction indices: up, down, left, right
    assert state[3:7].tolist() == [0, 0, 0, 1]


def test_danger_detects_wall():
    env = SnakeEnv(width=6, height=6, seed=42)

    env.snake = [(5, 3), (4, 3), (3, 3)]
    env.direction_index = 0

    state = encode_state(env)

    # Moving right would hit the wall
    assert state[0] == 1