"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from mltune.core.config import (
    Config,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    TuningConfig,
    SearchSpaceParam,
)


class TestSearchSpaceParam:
    """Tests for SearchSpaceParam."""
    
    def test_int_param(self):
        """Test integer parameter."""
        param = SearchSpaceParam(type="int", low=1, high=10)
        assert param.type == "int"
        assert param.low == 1
        assert param.high == 10
    
    def test_float_param(self):
        """Test float parameter."""
        param = SearchSpaceParam(type="float", low=0.0, high=1.0)
        assert param.type == "float"
    
    def test_loguniform_param(self):
        """Test loguniform parameter."""
        param = SearchSpaceParam(type="loguniform", low=1e-5, high=1e-1)
        assert param.type == "loguniform"
        assert param.log == False  # loguniform handles log internally
    
    def test_categorical_param(self):
        """Test categorical parameter."""
        param = SearchSpaceParam(type="categorical", choices=[1, 2, 3])
        assert param.type == "categorical"
        assert param.choices == [1, 2, 3]
    
    def test_invalid_param(self):
        """Test invalid parameter validation."""
        with pytest.raises(ValueError):
            SearchSpaceParam(type="int")  # Missing low/high


class TestExperimentConfig:
    """Tests for ExperimentConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ExperimentConfig(name="test")
        assert config.name == "test"
        assert config.task == "classification"
        assert config.direction == "minimize"
    
    def test_direction_validation(self):
        """Test direction validation."""
        config = ExperimentConfig(name="test", direction="maximize")
        assert config.direction == "maximize"
        
        with pytest.raises(ValueError):
            ExperimentConfig(name="test", direction="invalid")


class TestConfig:
    """Tests for Config class."""
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "experiment": {"name": "test_exp"},
            "training": {"epochs": 50},
        }
        config = Config.from_dict(data)
        
        assert config.experiment.name == "test_exp"
        assert config.training.epochs == 50
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        data = config.model_dump()
        
        assert "experiment" in data
        assert "model" in data
        assert "training" in data
    
    def test_yaml_roundtrip(self):
        """Test YAML save and load."""
        config = Config.from_dict({
            "experiment": {"name": "yaml_test"},
            "training": {"epochs": 100},
        })
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = Path(f.name)
        
        try:
            config.to_yaml(path)
            loaded = Config.from_yaml(path)
            
            assert loaded.experiment.name == "yaml_test"
            assert loaded.training.epochs == 100
        finally:
            path.unlink()
    
    def test_update(self):
        """Test config update."""
        config = Config()
        updated = config.update(**{"training.epochs": 200})
        
        assert updated.training.epochs == 200
    
    def test_env_substitution(self, monkeypatch):
        """Test environment variable substitution."""
        monkeypatch.setenv("TEST_LR", "0.001")
        
        yaml_content = """
experiment:
  name: test
training:
  learning_rate: "${TEST_LR}"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = Path(f.name)
        
        try:
            config = Config.from_yaml(path)
            assert config.training.learning_rate == "0.001"
        finally:
            path.unlink()
