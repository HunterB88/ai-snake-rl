"""Headless, reproducible comparison of trained DQN vs. random Snake.

Example: python evaluate.py --games 100 --seed 2026
Each policy sees the same sequence of environment seeds. Scores are measured
without exploration, gradient updates or any other training.
"""

import argparse
import csv
import random
from pathlib import Path
from statistics import mean, median

import torch

from model import DEFAULT_CHECKPOINT, load_trained_model
from snake_env import SnakeEnv
from state_encoder import encode_state


def play_episode(model, policy, seed):
    """Return (food eaten, moves taken) using a fixed episode seed."""
    if policy not in ("RANDOM", "TRAINED"):
        raise ValueError("Policy must be RANDOM or TRAINED")

    env = SnakeEnv(seed=seed)
    rng = random.Random(seed + 10_000)
    moves = 0
    steps_without_food = 0

    while not env.done:
        if policy == "RANDOM":
            action = rng.randrange(3)
        else:
            state = encode_state(env)
            with torch.inference_mode():
                action = int(model(state.unsqueeze(0)).argmax(dim=1).item())

        _, reward, ended = env.step(action)
        moves += 1
        steps_without_food = 0 if reward == 10.0 else steps_without_food + 1
        # Match the training/demo no-food limit so looping is not rewarded.
        if ended or steps_without_food >= 100 * len(env.snake):
            break

    return env.score, moves


def evaluate(model, games=100, seed=2026):
    """Return independent paired-seed results; never update model weights."""
    if games < 1:
        raise ValueError("games must be at least 1")
    if model is None:
        raise ValueError("A trained model is required")

    model.eval()
    results = []
    for policy in ("RANDOM", "TRAINED"):
        for episode in range(games):
            score, steps = play_episode(model, policy, seed + episode)
            results.append({
                "policy": policy,
                "episode": episode + 1,
                "seed": seed + episode,
                "score": score,
                "steps": steps,
            })
    return results


def save_results(results, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("policy", "episode", "seed", "score", "steps"))
        writer.writeheader()
        writer.writerows(results)
    return path


def summarize(results):
    """Summaries are based on evaluation episodes, not training metrics."""
    summary = {}
    for policy in ("RANDOM", "TRAINED"):
        scores = [row["score"] for row in results if row["policy"] == policy]
        summary[policy] = {
            "games": len(scores),
            "mean": mean(scores),
            "median": median(scores),
            "best": max(scores),
            "food_rate": sum(score > 0 for score in scores) / len(scores),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100,
                        help="Number of evaluation games per policy (default: 100)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Starting environment seed (default: 2026)")
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to trained model state_dict")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.csv"),
                        help="CSV destination")
    args = parser.parse_args()

    model = load_trained_model(args.model)
    print(f"Evaluating {args.games} paired seeds with no exploration...", flush=True)
    results = evaluate(model, games=args.games, seed=args.seed)
    summary = summarize(results)
    print("\nPOLICY   GAMES   MEAN   MEDIAN   BEST   ATE FOOD")
    for policy in ("RANDOM", "TRAINED"):
        row = summary[policy]
        print(f"{policy:8} {row['games']:5d}   {row['mean']:5.2f}   "
              f"{row['median']:6.2f}   {row['best']:4d}   {row['food_rate']:.0%}")
    print(f"\nResults saved to {save_results(results, args.output)}")


if __name__ == "__main__":
    main()
