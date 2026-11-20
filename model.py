from pathlib import Path

import torch
import torch.nn as nn


class SnakeQNetwork(nn.Module):
    """Deep Q-Network for the Snake agent."""

    def __init__(self):
        super().__init__()

        # Neural network layers
        self.input_layer = nn.Linear(11, 128)

        self.hidden_layer = nn.Linear(128, 128)

        self.output_layer = nn.Linear(128, 3)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x, return_activations=False):
        """Perform forward propagation."""

        # First hidden layer
        hidden_1 = self.relu(self.input_layer(x))

        # Second hidden layer
        hidden_2 = self.relu(self.hidden_layer(hidden_1))

        # Output layer
        q_values = self.output_layer(hidden_2)

        # Save activations for our future visualization
        if return_activations:
            activations = {
                "hidden_1": hidden_1,
                "hidden_2": hidden_2,
            }

            return q_values, activations

        return q_values


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "snake_dqn.pt"


def load_trained_model(path=DEFAULT_CHECKPOINT):
    """Load our own saved state dictionary for inference on the CPU."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {path}")
    network = SnakeQNetwork()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return network.eval()
