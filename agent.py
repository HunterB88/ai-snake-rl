import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from model import SnakeQNetwork


class ReplayBuffer:
    """Stores previous experiences for training."""

    def __init__(self, capacity=50000, seed=None):
        self.memory = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        """Save one experience."""

        experience = (
            state.detach().clone(),
            action,
            reward,
            next_state.detach().clone(),
            done,
        )

        self.memory.append(experience)

    def sample(self, batch_size):
        """Sample random experiences."""

        return self.rng.sample(list(self.memory), batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Deep Q-Learning agent for Snake."""

    def __init__(
        self,
        learning_rate=0.001,
        gamma=0.95,
        batch_size=128,
        seed=None,
    ):

        self.rng = random.Random(seed)

        # Main neural network
        self.policy_net = SnakeQNetwork()

        # Target neural network
        self.target_net = SnakeQNetwork()

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        # Loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Experience replay
        self.memory = ReplayBuffer(seed=seed)

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size

        # Exploration settings
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Target network updates
        self.learn_steps = 0
        self.target_update_frequency = 250

    def choose_action(self, state, training=True):
        """Choose an action using epsilon-greedy exploration."""

        # Explore with a random action
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(3)

        # Otherwise, use the neural network
        with torch.no_grad():

            q_values = self.policy_net(
                state.unsqueeze(0)
            )

            action = torch.argmax(
                q_values,
                dim=1,
            ).item()

        return int(action)

    def remember(self, state, action, reward, next_state, done):
        """Store an experience."""

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def learn(self):
        """Train the neural network using experience replay."""

        if len(self.memory) < self.batch_size:
            return None

        # Sample experiences
        batch = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)

        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Current predicted Q-values
        current_q = self.policy_net(states)

        selected_q = current_q.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # Calculate target Q-values
        with torch.no_grad():

            next_q = self.target_net(
                next_states
            ).max(dim=1).values

            target_q = rewards + (
                self.gamma * next_q * (1 - dones)
            )

        # Calculate loss
        loss = self.loss_fn(selected_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        # Prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.learn_steps += 1

        # Update target network periodically
        if self.learn_steps % self.target_update_frequency == 0:

            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return loss.item()

    def end_episode(self):
        """Reduce exploration after each game."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )