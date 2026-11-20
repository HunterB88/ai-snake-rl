import torch

from model import SnakeQNetwork


def test_network_output_shape():
    model = SnakeQNetwork()

    # Batch of four game states
    states = torch.zeros((4, 11))

    output = model(states)

    assert output.shape == (4, 3)


def test_hidden_layer_activations():
    model = SnakeQNetwork()

    state = torch.zeros((1, 11))

    q_values, activations = model(
        state,
        return_activations=True,
    )

    assert q_values.shape == (1, 3)
    assert activations["hidden_1"].shape == (1, 128)
    assert activations["hidden_2"].shape == (1, 128)


def test_network_supports_backpropagation():
    model = SnakeQNetwork()

    state = torch.zeros((1, 11))

    output = model(state)

    loss = output.sum()

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )