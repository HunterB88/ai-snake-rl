import torch

from agent import DQNAgent, ReplayBuffer


def test_replay_buffer_capacity():
    memory = ReplayBuffer(capacity=2)

    state = torch.zeros(11)

    for i in range(5):
        memory.push(
            state,
            0,
            1.0,
            state,
            False,
        )

    assert len(memory) == 2


def test_agent_returns_valid_action():
    agent = DQNAgent(seed=42)

    state = torch.zeros(11)

    action = agent.choose_action(state)

    assert action in (0, 1, 2)


def test_agent_requires_enough_experiences():
    agent = DQNAgent(batch_size=4)

    loss = agent.learn()

    assert loss is None


def test_agent_can_learn():
    agent = DQNAgent(batch_size=4, seed=42)

    state = torch.zeros(11)
    next_state = torch.ones(11)

    # Store four experiences
    for _ in range(4):
        agent.remember(
            state,
            0,
            1.0,
            next_state,
            False,
        )

    # Save weights before training
    old_weights = [
        parameter.detach().clone()
        for parameter in agent.policy_net.parameters()
    ]

    # Perform one training step
    loss = agent.learn()

    # Confirm training occurred
    assert loss is not None
    assert loss >= 0

    # Confirm at least one parameter changed
    changed = any(
        not torch.equal(old, new)
        for old, new in zip(
            old_weights,
            agent.policy_net.parameters(),
        )
    )

    assert changed