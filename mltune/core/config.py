"""
Configuration management system for AutoForge.

Provides YAML/JSON based configuration with:
- Configuration inheritance
- Parameter validation
- Environment variable substitution
- Dynamic value computation
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Type alias for type hints
T = TypeVar("T")


class SearchSpaceParam(BaseModel):
    """Definition of a hyperparameter search space."""
    
    type: str = Field(..., description="Parameter type: int, float, categorical, loguniform")
    low: Optional[Union[int, float]] = Field(None, description="Lower bound for numeric types")
    high: Optional[Union[int, float]] = Field(None, description="Upper bound for numeric types")
    choices: Optional[List[Any]] = Field(None, description="Choices for categorical type")
    step: Optional[Union[int, float]] = Field(None, description="Step size for discrete parameters")
    log: bool = Field(False, description="Whether to use log scale")
    
    @model_validator(mode="after")
    def validate_param(self) -> "SearchSpaceParam":
        """Validate parameter definition based on type."""
        if self.type in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(f"'{self.type}' type requires 'low' and 'high' bounds")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError("'categorical' type requires 'choices' list")
        elif self.type == "loguniform":
            if self.low is None or self.high is None:
                raise ValueError("'loguniform' type requires 'low' and 'high' bounds")
            if self.low <= 0 or self.high <= 0:
                raise ValueError("'loguniform' bounds must be positive")
        return self


class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    
    name: str = Field("default_experiment", description="Experiment name")
    task: str = Field("classification", description="Task type: classification, regression, generation")
    objective: str = Field("val_loss", description="Optimization objective metric")
    direction: str = Field("minimize", description="Optimization direction: minimize or maximize")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")
    description: Optional[str] = Field(None, description="Experiment description")
    
    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        return v


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    
    architecture: str = Field("transformer", description="Model architecture type")
    num_layers: int = Field(12, ge=1, description="Number of layers")
    hidden_dim: int = Field(512, ge=1, description="Hidden dimension size")
    num_heads: int = Field(8, ge=1, description="Number of attention heads")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
    activation: str = Field("gelu", description="Activation function")
    vocab_size: Optional[int] = Field(None, description="Vocabulary size for language models")
    max_seq_len: Optional[int] = Field(None, description="Maximum sequence length")
    
    # Additional architecture-specific params
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""
    
    epochs: int = Field(100, ge=1, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, description="Batch size")
    learning_rate: float = Field(1e-3, gt=0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay coefficient")
    optimizer: str = Field("adamw", description="Optimizer type")
    scheduler: str = Field("cosine", description="Learning rate scheduler")
    warmup_steps: int = Field(0, ge=0, description="Number of warmup steps")
    gradient_clip: Optional[float] = Field(None, ge=0, description="Gradient clipping threshold")
    accumulation_steps: int = Field(1, ge=1, description="Gradient accumulation steps")
    
    # Early stopping
    early_stopping: bool = Field(False, description="Enable early stopping")
    patience: int = Field(10, ge=1, description="Early stopping patience")
    min_delta: float = Field(0.0, ge=0, description="Minimum improvement for early stopping")


class TuningConfig(BaseModel):
    """Hyperparameter tuning configuration."""
    
    strategy: str = Field("bayesian", description="Optimization strategy")
    n_trials: int = Field(50, ge=1, description="Number of optimization trials")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    n_jobs: int = Field(1, description="Number of parallel jobs")
    search_space: Dict[str, SearchSpaceParam] = Field(
        default_factory=dict, 
        description="Hyperparameter search space"
    )
    pruning: bool = Field(True, description="Enable trial pruning")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid_strategies = ("bayesian", "tpe", "cmaes", "grid", "random", "agent")
        if v not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        return v


class DataConfig(BaseModel):
    """Data configuration."""
    
    train_path: Optional[str] = Field(None, description="Training data path")
    val_path: Optional[str] = Field(None, description="Validation data path")
    test_path: Optional[str] = Field(None, description="Test data path")
    train_split: float = Field(0.8, gt=0, lt=1, description="Training split ratio")
    num_workers: int = Field(4, ge=0, description="DataLoader workers")
    pin_memory: bool = Field(True, description="Pin memory for faster GPU transfer")


class LoggingConfig(BaseModel):
    """Logging and checkpointing configuration."""
    
    log_dir: str = Field("logs", description="Log directory")
    checkpoint_dir: str = Field("checkpoints", description="Checkpoint directory")
    save_best_only: bool = Field(True, description="Save only best model")
    log_interval: int = Field(100, description="Logging interval (steps)")
    save_interval: int = Field(1, description="Checkpoint save interval (epochs)")
    
    # External loggers
    use_tensorboard: bool = Field(True, description="Use TensorBoard")
    use_wandb: bool = Field(False, description="Use Weights & Biases")
    wandb_project: Optional[str] = Field(None, description="W&B project name")


class DistributedConfig(BaseModel):
    """Distributed training configuration."""
    
    enabled: bool = Field(False, description="Enable distributed training")
    backend: str = Field("nccl", description="Distributed backend")
    strategy: str = Field("ddp", description="Distribution strategy: ddp, deepspeed, fsdp")
    num_gpus: int = Field(1, ge=1, description="Number of GPUs")
    num_nodes: int = Field(1, ge=1, description="Number of nodes")


class Config(BaseModel):
    """
    Main configuration class that aggregates all configuration sections.
    
    Supports:
    - YAML/JSON loading
    - Configuration inheritance
    - Environment variable substitution
    - Validation
    """
    
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    
    # Internal state
    _config_path: Optional[Path] = None
    _parent_config: Optional["Config"] = None
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path], **overrides: Any) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            **overrides: Parameter overrides
            
        Returns:
            Config instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Handle inheritance
        if "_base_" in data:
            base_path = path.parent / data.pop("_base_")
            base_config = cls.from_yaml(base_path)
            data = cls._deep_merge(base_config.model_dump(), data)
        
        # Handle environment variable substitution
        data = cls._substitute_env_vars(data)
        
        # Apply overrides
        data = cls._deep_merge(data, overrides)
        
        config = cls(**data)
        config._config_path = path
        return config
    
    @classmethod
    def from_json(cls, path: Union[str, Path], **overrides: Any) -> "Config":
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data = cls._deep_merge(data, overrides)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **overrides: Any) -> "Config":
        """Create configuration from dictionary."""
        data = cls._deep_merge(data, overrides)
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump(exclude_none=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(exclude_none=True), f, indent=2)
    
    def update(self, **kwargs: Any) -> "Config":
        """
        Update configuration parameters.
        
        Supports dot notation for nested updates:
            config.update(**{"model.num_layers": 24, "training.learning_rate": 0.0001})
        """
        data = self.model_dump()
        for key, value in kwargs.items():
            keys = key.split(".")
            d = data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        return Config.from_dict(data)
    
    def validate_config(self) -> bool:
        """Validate the entire configuration."""
        # Pydantic validates on construction, this is for additional checks
        if self.training.warmup_steps > self.training.epochs * 1000:
            print("Warning: warmup_steps seems too large relative to epochs")
        return True
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    
    @staticmethod
    def _substitute_env_vars(data: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {k: Config._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Match ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2)
                value = os.environ.get(var_name, default)
                if value is None:
                    raise ValueError(f"Environment variable '{var_name}' not set and no default provided")
                return value
            
            return re.sub(pattern, replacer, data)
        return data
    
    def __repr__(self) -> str:
        return f"Config(experiment={self.experiment.name}, task={self.experiment.task})"
