"""Tests for the real 270-neuron network visualization."""

import pygame
import torch

from model import SnakeQNetwork
from network_explorer import NetworkExplorer
from snake_env import SnakeEnv
from state_encoder import encode_state


def test_all_270_neurons_and_all_18176_connections():
    explorer = NetworkExplorer(SnakeQNetwork())

    assert [
        len(layer)
        for layer in explorer.positions()
    ] == [11, 128, 128, 3]

    total_connections = sum(
        len(row)
        for matrix in explorer.weights
        for row in matrix
    )

    assert total_connections == 18_176


def test_explorer_displays_real_forward_pass():
    pygame.font.init()

    model = SnakeQNetwork().eval()
    explorer = NetworkExplorer(model)

    screen = pygame.Surface((1220, 748))

    state = encode_state(SnakeEnv(seed=19))

    with torch.inference_mode():
        q_values, hidden = model(
            state.unsqueeze(0),
            return_activations=True,
        )

    inputs = state.tolist()
    hidden_1 = hidden["hidden_1"][0].tolist()
    hidden_2 = hidden["hidden_2"][0].tolist()
    outputs = q_values[0].tolist()

    explorer.draw(
        screen,
        inputs,
        hidden_1,
        hidden_2,
        outputs,
    )

    assert screen.get_at((100, 100)) is not None

    actual_activations = explorer.activations(
        inputs,
        hidden_1,
        hidden_2,
        outputs,
    )

    assert actual_activations[3] == outputs


def test_select_neuron_and_toggle_all_weights():
    pygame.font.init()

    explorer = NetworkExplorer(SnakeQNetwork())
    screen = pygame.Surface((1220, 748))

    inputs = [0] * 11
    hidden_1 = [0] * 128
    hidden_2 = [0] * 128
    outputs = [0] * 3

    explorer.draw(
        screen,
        inputs,
        hidden_1,
        hidden_2,
        outputs,
    )

    # Select neuron 61 in the first hidden layer.
    point = explorer.positions()[1][60]

    explorer.click(point)

    assert explorer.selected == (1, 60)

    # Enable visualization of all learned weights.
    explorer.click(
        explorer.buttons["all"].center
    )

    assert explorer.show_all is True

    # Disable visualization of all learned weights.
    explorer.draw(
        screen,
        inputs,
        hidden_1,
        hidden_2,
        outputs,
    )

    explorer.click(
        explorer.buttons["all"].center
    )

    assert explorer.show_all is False


def test_older_demo_mouse_handlers_do_not_crash():
    explorer = NetworkExplorer(SnakeQNetwork())

    # Compatibility with older versions of the demo.
    explorer.drag((300, 400))
    explorer.wheel(1)

    explorer.dragging = False