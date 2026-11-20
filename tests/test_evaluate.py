"""Headless, deterministic checks for the published evaluation workflow."""

import csv

import pytest

from evaluate import evaluate, play_episode, save_results, summarize
from model import SnakeQNetwork


def test_evaluation_compares_equal_seeds_and_exports(tmp_path):
    model = SnakeQNetwork()
    results = evaluate(model, games=2, seed=91)

    assert len(results) == 4
    assert [row["seed"] for row in results] == [91, 92, 91, 92]
    assert [row["policy"] for row in results] == ["RANDOM"] * 2 + ["TRAINED"] * 2
    assert all(row["score"] >= 0 for row in results)
    assert all(row["steps"] > 0 for row in results)

    path = save_results(results, tmp_path / "runs" / "evaluation.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 4
    assert {row["policy"] for row in exported} == {"RANDOM", "TRAINED"}
    assert set(summarize(results)) == {"RANDOM", "TRAINED"}


def test_random_policy_is_repeatable():
    first = play_episode(None, "RANDOM", 123)
    assert first == play_episode(None, "RANDOM", 123)


def test_evaluation_validates_inputs():
    with pytest.raises(ValueError):
        evaluate(SnakeQNetwork(), games=0)
    with pytest.raises(ValueError):
        evaluate(None, games=1)
    with pytest.raises(ValueError):
        play_episode(None, "UNKNOWN", seed=12)
