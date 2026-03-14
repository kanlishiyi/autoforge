"""Tests for experiment tracking."""

import tempfile
from pathlib import Path

from mltune.core.config import Config
from mltune.core.experiment import Experiment


class TestExperiment:
    """Tests for Experiment class."""

    def test_create_experiment(self):
        """Test experiment creation."""
        exp = Experiment("test_experiment")

        assert exp.name == "test_experiment"
        assert exp.experiment_id.startswith("exp_")

    def test_track_context(self):
        """Test experiment tracking context."""
        exp = Experiment("test")

        with exp.track():
            exp.log_metric("train_loss", 0.5, step=1)
            exp.log_metric("train_loss", 0.3, step=2)

        metrics = exp.get_metrics("train_loss")
        assert len(metrics) == 2

    def test_metric_logging(self):
        """Test metric logging."""
        exp = Experiment("test")
        exp.start()

        exp.log_metric("accuracy", 0.8, step=1)
        exp.log_metric("accuracy", 0.85, step=2)
        exp.log_metric("loss", 0.5, step=1)

        history = exp.get_metric_history("accuracy")
        assert len(history) == 2
        assert history[0] == 0.8
        assert history[1] == 0.85

    def test_best_metric(self):
        """Test best metric tracking."""
        config = Config.from_dict({
            "experiment": {
                "name": "test",
                "objective": "val_loss",
                "direction": "minimize",
            },
        })

        exp = Experiment("test", config=config)
        exp.start()

        exp.log_metric("val_loss", 0.5, step=1)
        exp.log_metric("val_loss", 0.3, step=2)
        exp.log_metric("val_loss", 0.4, step=3)

        best = exp.get_best_metric("val_loss")
        assert best == 0.3

    def test_save_and_load(self):
        """Test experiment save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment("test", storage_dir=tmpdir)

            with exp.track():
                exp.log_metric("loss", 0.5, step=1)
                exp.log_metric("accuracy", 0.8, step=1)

            path = exp.save()

            loaded = Experiment.load(path)
            assert loaded.name == "test"
            assert len(loaded.get_metrics()) == 2

    def test_artifact_logging(self):
        """Test artifact logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "model.pt"
            test_file.write_text("test")

            exp = Experiment("test")
            exp.log_artifact(test_file, artifact_type="model")

            assert len(exp._state.artifacts) == 1

    def test_experiment_comparison(self):
        """Test experiment comparison."""
        exp1 = Experiment("exp1")
        exp2 = Experiment("exp2")

        with exp1.track():
            exp1.log_metric("accuracy", 0.8, step=1)

        with exp2.track():
            exp2.log_metric("accuracy", 0.9, step=1)

        comparison = Experiment.compare([exp1, exp2], metrics=["accuracy"])
        assert "accuracy" in comparison["metrics"]
