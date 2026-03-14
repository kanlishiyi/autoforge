"""
Experiment management and tracking system.

Provides comprehensive experiment lifecycle management including:
- Experiment creation and configuration
- Metric logging and tracking
- Artifact management
- Experiment comparison and analysis
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from mltune.core.config import Config


class MetricRecord(BaseModel):
    """A single metric record."""
    
    name: str
    value: float
    step: int
    timestamp: float = Field(default_factory=time.time)
    context: Dict[str, Any] = Field(default_factory=dict)


class ArtifactInfo(BaseModel):
    """Information about an artifact."""
    
    name: str
    path: str
    type: str
    size_bytes: int
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentState(BaseModel):
    """Experiment state information."""
    
    experiment_id: str
    name: str
    status: str = "created"  # created, running, completed, failed, stopped
    config: Dict[str, Any]
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    metrics: List[MetricRecord] = Field(default_factory=list)
    artifacts: List[ArtifactInfo] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    best_metric: Optional[float] = None
    best_step: Optional[int] = None


class Experiment:
    """
    Experiment management class.
    
    Provides comprehensive tracking and management of ML experiments:
    
    Example:
        ```python
        config = Config.from_yaml("config.yaml")
        exp = Experiment("my_experiment", config=config)
        
        with exp.track():
            for epoch in range(100):
                loss = train_epoch()
                exp.log_metric("train_loss", loss, step=epoch)
                
                val_loss = validate()
                exp.log_metric("val_loss", val_loss, step=epoch)
        
        # Save and analyze
        exp.save()
        exp.summary()
        ```
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Config] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        storage_dir: Union[str, Path] = "experiments",
        experiment_id: Optional[str] = None,
    ):
        """
        Initialize an experiment.
        
        Args:
            name: Experiment name
            config: Configuration object
            tags: List of tags for organization
            parent_id: Parent experiment ID for hierarchical experiments
            storage_dir: Directory to store experiment data
            experiment_id: Custom experiment ID (auto-generated if None)
        """
        self.experiment_id = experiment_id or self._generate_id()
        self.name = name
        self.config = config
        self.parent_id = parent_id
        self.storage_dir = Path(storage_dir)
        
        # Initialize state
        self._state = ExperimentState(
            experiment_id=self.experiment_id,
            name=name,
            config=config.model_dump() if config else {},
            tags=tags or [],
            parent_id=parent_id,
        )
        
        # Tracking state
        self._is_tracking = False
        self._step_counter: Dict[str, int] = {}
        self._metric_history: Dict[str, List[float]] = {}
        
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def _generate_id() -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"exp_{timestamp}_{unique}"
    
    def start(self) -> "Experiment":
        """Start the experiment."""
        if self._state.status == "created":
            self._state.status = "running"
            self._state.started_at = time.time()
        return self
    
    def stop(self, status: str = "completed") -> "Experiment":
        """Stop the experiment."""
        self._state.status = status
        self._state.ended_at = time.time()
        self._is_tracking = False
        return self
    
    @contextmanager
    def track(self):
        """
        Context manager for automatic experiment tracking.
        
        Yields:
            Experiment instance
        """
        self.start()
        self._is_tracking = True
        try:
            yield self
            self.stop("completed")
        except Exception as e:
            self.stop("failed")
            self.log_metadata("error", str(e))
            raise
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        **context: Any,
    ) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (auto-incremented if None)
            **context: Additional context for this metric
        """
        if step is None:
            step = self._step_counter.get(name, 0)
            self._step_counter[name] = step + 1
        
        record = MetricRecord(
            name=name,
            value=value,
            step=step,
            context=context,
        )
        
        self._state.metrics.append(record)
        
        # Track best metric
        metric_history = self._metric_history.setdefault(name, [])
        metric_history.append(value)
        
        if self.config:
            objective = self.config.experiment.objective
            direction = self.config.experiment.direction
            
            if name == objective:
                is_best = False
                if self._state.best_metric is None:
                    is_best = True
                elif direction == "minimize" and value < self._state.best_metric:
                    is_best = True
                elif direction == "maximize" and value > self._state.best_metric:
                    is_best = True
                
                if is_best:
                    self._state.best_metric = value
                    self._state.best_step = step
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number for all metrics
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)
    
    def log_artifact(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
        **metadata: Any,
    ) -> ArtifactInfo:
        """
        Log an artifact (file).
        
        Args:
            path: Path to the artifact file
            name: Artifact name (filename if None)
            artifact_type: Type of artifact (model, checkpoint, log, etc.)
            **metadata: Additional metadata
            
        Returns:
            ArtifactInfo object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        
        artifact = ArtifactInfo(
            name=name or path.name,
            path=str(path),
            type=artifact_type,
            size_bytes=path.stat().st_size,
            metadata=metadata,
        )
        
        self._state.artifacts.append(artifact)
        return artifact
    
    def log_metadata(self, key: str, value: Any) -> None:
        """Log arbitrary metadata."""
        # Store in config dict for persistence
        self._state.config[f"_metadata_{key}"] = value
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self._state.config["hyperparameters"] = params
    
    def get_metrics(self, name: Optional[str] = None) -> List[MetricRecord]:
        """
        Get logged metrics.
        
        Args:
            name: Filter by metric name (all if None)
            
        Returns:
            List of MetricRecord objects
        """
        if name:
            return [m for m in self._state.metrics if m.name == name]
        return self._state.metrics.copy()
    
    def get_metric_history(self, name: str) -> List[float]:
        """Get history of a specific metric."""
        return self._metric_history.get(name, [])
    
    def get_best_metric(self, name: Optional[str] = None) -> Optional[float]:
        """Get best value for a metric."""
        if name is None:
            return self._state.best_metric
        
        history = self.get_metric_history(name)
        if not history:
            return None
        
        if self.config:
            direction = self.config.experiment.direction
            if direction == "minimize":
                return min(history)
            else:
                return max(history)
        return history[-1]
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate experiment summary.
        
        Returns:
            Summary dictionary
        """
        duration = None
        if self._state.started_at:
            end = self._state.ended_at or time.time()
            duration = end - self._state.started_at
        
        metric_summaries = {}
        for name, history in self._metric_history.items():
            if history:
                metric_summaries[name] = {
                    "min": min(history),
                    "max": max(history),
                    "mean": sum(history) / len(history),
                    "last": history[-1],
                    "count": len(history),
                }
        
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self._state.status,
            "duration_seconds": duration,
            "best_metric": self._state.best_metric,
            "best_step": self._state.best_step,
            "metrics": metric_summaries,
            "artifacts": len(self._state.artifacts),
            "tags": self._state.tags,
        }
    
    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save experiment state to disk.
        
        Args:
            path: Save path (default: storage_dir/experiment_id/state.json)
            
        Returns:
            Path to saved file
        """
        if path is None:
            path = self.storage_dir / self.experiment_id / "state.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._state.model_dump(), f, indent=2, default=str)
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path], storage_dir: Optional[Path] = None) -> "Experiment":
        """
        Load experiment from disk.
        
        Args:
            path: Path to experiment state file
            storage_dir: Storage directory (derived from path if None)
            
        Returns:
            Experiment instance
        """
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        state = ExperimentState(**data)
        
        exp = cls(
            name=state.name,
            experiment_id=state.experiment_id,
            storage_dir=storage_dir or path.parent.parent,
        )
        exp._state = state
        
        # Rebuild metric history
        for record in state.metrics:
            history = exp._metric_history.setdefault(record.name, [])
            history.append(record.value)
        
        return exp
    
    @staticmethod
    def compare(experiments: List["Experiment"], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiments: List of experiments to compare
            metrics: Metrics to compare (all metrics if None)
            
        Returns:
            Comparison dictionary
        """
        if not experiments:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp._metric_history.keys())
        
        metrics = metrics or list(all_metrics)
        
        comparison = {
            "experiments": [exp.experiment_id for exp in experiments],
            "metrics": {},
        }
        
        for metric in metrics:
            values = []
            for exp in experiments:
                history = exp._metric_history.get(metric, [])
                values.append({
                    "experiment_id": exp.experiment_id,
                    "best": exp.get_best_metric(metric),
                    "last": history[-1] if history else None,
                    "count": len(history),
                })
            comparison["metrics"][metric] = values
        
        return comparison
    
    def __repr__(self) -> str:
        return f"Experiment(id={self.experiment_id}, name={self.name}, status={self._state.status})"
