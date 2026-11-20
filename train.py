import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from agent import DQNAgent
from snake_env import SnakeEnv
from state_encoder import encode_state


def train(
    episodes=100,
    seed=42,
    output_root=Path("training_output"),
    save_plot=True,
):
    """Train the Snake AI using Deep Q-Learning."""

    if episodes < 1:
        raise ValueError("Episodes must be at least 1")

    torch.manual_seed(seed)

    # Small networks often train efficiently with one CPU thread
    torch.set_num_threads(1)

    output_root = Path(output_root)

    model_dir = output_root / "models"
    runs_dir = output_root / "runs"

    model_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create environment and agent
    env = SnakeEnv(seed=seed)
    agent = DQNAgent(seed=seed)

    scores = []
    average_scores = []
    epsilon_history = []
    training_log = []

    print("\nStarting Snake AI training...\n", flush=True)

    for episode in range(1, episodes + 1):

        # Start a new game
        env.reset()

        done = False
        steps = 0
        steps_without_food = 0

        episode_losses = []

        while not done:

            # Observe current state
            state = encode_state(env)

            # Choose action
            action = agent.choose_action(
                state,
                training=True,
            )

            # Execute action
            _, reward, done = env.step(action)

            steps += 1

            # Track how long the snake goes without eating
            if reward == 10.0:
                steps_without_food = 0
            else:
                steps_without_food += 1

            # Prevent the snake from circling forever
            if (
                not done
                and steps_without_food >= 100 * len(env.snake)
            ):
                done = True
                reward = -10.0

            # Observe the next state
            next_state = encode_state(env)

            # Store experience
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # Train neural network
            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

        # Reduce exploration
        agent.end_episode()

        # Record score
        scores.append(env.score)

        avg_score = sum(scores[-50:]) / len(scores[-50:])

        average_scores.append(avg_score)
        epsilon_history.append(agent.epsilon)

        avg_loss = (
            sum(episode_losses) / len(episode_losses)
            if episode_losses
            else 0.0
        )

        training_log.append({
            "episode": episode,
            "score": env.score,
            "steps": steps,
            "average_score_50": avg_score,
            "epsilon": agent.epsilon,
            "average_loss": avg_loss,
        })

        # Print progress
        if episode == 1 or episode % 10 == 0 or episode == episodes:

            print(
                f"Episode {episode}/{episodes} | "
                f"Score: {env.score} | "
                f"Avg Score: {avg_score:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}",
                flush=True,
            )

    # Save trained neural network weights
    model_path = model_dir / "snake_dqn.pt"

    torch.save(
        agent.policy_net.state_dict(),
        model_path,
    )

    # Save training data
    csv_path = runs_dir / "training_metrics.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=training_log[0].keys(),
        )

        writer.writeheader()
        writer.writerows(training_log)

    # Generate training visualization
    if save_plot:

        plt.style.use("dark_background")

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
        )

        episode_numbers = range(1, episodes + 1)

        # Score graph
        ax1.plot(
            episode_numbers,
            scores,
            color="gray",
            alpha=0.35,
            label="Individual Score",
        )

        ax1.plot(
            episode_numbers,
            average_scores,
            color="#00ff88",
            linewidth=2,
            label="Average Score (Last 50)",
        )

        ax1.set_title("Snake AI - Training Progress")
        ax1.set_ylabel("Score")
        ax1.legend()
        ax1.grid(alpha=0.2)

        # Exploration graph
        ax2.plot(
            episode_numbers,
            epsilon_history,
            color="#bb86fc",
            linewidth=2,
        )

        ax2.set_xlabel("Training Episode")
        ax2.set_ylabel("Exploration Rate")
        ax2.grid(alpha=0.2)

        plt.tight_layout()

        graph_path = runs_dir / "training_progress.png"

        plt.savefig(
            graph_path,
            dpi=150,
        )

        plt.close(fig)

    print("\nTraining complete!")
    print(f"Final average score: {average_scores[-1]:.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Model saved to: {model_path}")
    print(f"Training data saved to: {csv_path}")

    if save_plot:
        print(f"Graph saved to: {runs_dir / 'training_progress.png'}")

    return training_log


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the Snake DQN agent"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_output"),
        help="Directory for new models and graphs; preserves the demo checkpoint",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip creating a progress PNG",
    )

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        seed=args.seed,
        output_root=args.output_root,
        save_plot=not args.no_plot,
    )
