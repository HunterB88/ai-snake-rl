"""The included checkpoint must load and produce three action values."""

from pathlib import Path

import torch

from model import DEFAULT_CHECKPOINT, load_trained_model
from snake_env import SnakeEnv
from state_encoder import encode_state


def test_bundled_checkpoint_is_inference_ready():
    assert DEFAULT_CHECKPOINT.is_file()
    model = load_trained_model()
    with torch.inference_mode():
        values = model(encode_state(SnakeEnv(seed=13)).unsqueeze(0))
    assert values.shape == (1, 3)
    assert torch.isfinite(values).all()


def test_checkpoint_missing_path_has_helpful_error(tmp_path):
    missing = tmp_path / "missing.pt"
    try:
        load_trained_model(missing)
    except FileNotFoundError as exc:
        assert str(Path(missing)) in str(exc)
    else:
        raise AssertionError("Missing model must raise FileNotFoundError")
