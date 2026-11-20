from train import train


def test_short_training(tmp_path):

    results = train(
        episodes=2,
        seed=42,
        output_root=tmp_path,
        save_plot=False,
    )

    assert len(results) == 2

    assert (tmp_path / "models" / "snake_dqn.pt").exists()

    assert (tmp_path / "runs" / "training_metrics.csv").exists()