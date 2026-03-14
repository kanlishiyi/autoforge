"""
Study class for storing and analyzing optimization results.

A Study contains:
- All trial results
- Best parameters
- Optimization history
- Analysis methods
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from mltune.core.config import Config
from mltune.optim.base import TrialResult, TrialState


class StudySummary(BaseModel):
    """Summary of a study."""
    
    study_name: str
    direction: str
    n_trials: int
    n_completed_trials: int
    n_failed_trials: int
    best_value: Optional[float]
    best_params: Optional[Dict[str, Any]]
    best_model_path: Optional[str] = None
    duration_seconds: Optional[float]
    created_at: str
    updated_at: str


class Study:
    """
    A Study collects all trials and provides analysis methods.
    
    Example:
        ```python
        study = tuner.optimize(n_trials=100)
        
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_params}")
        
        # Analysis
        study.param_importance()
        study.plot_optimization_history()
        study.plot_slice()
        ```
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        direction: str = "minimize",
        study_name: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize a Study.
        
        Args:
            config: Configuration object
            direction: Optimization direction ("minimize" or "maximize")
            study_name: Study name (auto-generated if None)
            storage_path: Path to store study data
        """
        self.config = config
        self.direction = direction
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._trials: List[TrialResult] = []
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._best_model_path: Optional[str] = None
    
    def add_trial(self, trial: TrialResult) -> None:
        """Add a trial result to the study."""
        self._trials.append(trial)
        self._updated_at = datetime.now()
    
    @property
    def trials(self) -> List[TrialResult]:
        """Get all trials."""
        return self._trials.copy()
    
    @property
    def best_trial(self) -> Optional[TrialResult]:
        """Get the best trial."""
        completed = [
            t for t in self._trials 
            if t.state == TrialState.COMPLETED and t.value is not None
        ]
        
        if not completed:
            return None
        
        if self.direction == "minimize":
            return min(completed, key=lambda t: t.value)
        else:
            return max(completed, key=lambda t: t.value)
    
    @property
    def best_value(self) -> Optional[float]:
        """Get the best objective value."""
        trial = self.best_trial
        return trial.value if trial else None
    
    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters."""
        trial = self.best_trial
        return trial.params if trial else None

    @property
    def best_model_path(self) -> Optional[str]:
        """Get the path to the saved best model artifact."""
        return self._best_model_path

    @best_model_path.setter
    def best_model_path(self, path: Optional[str]) -> None:
        """Set the path to the saved best model artifact."""
        self._best_model_path = path
        self._updated_at = datetime.now()
    
    @property
    def n_trials(self) -> int:
        """Get total number of trials."""
        return len(self._trials)
    
    @property
    def n_completed(self) -> int:
        """Get number of completed trials."""
        return sum(1 for t in self._trials if t.state == TrialState.COMPLETED)
    
    @property
    def n_failed(self) -> int:
        """Get number of failed trials."""
        return sum(1 for t in self._trials if t.state == TrialState.FAILED)
    
    def get_trials_by_state(self, state: TrialState) -> List[TrialResult]:
        """Get trials filtered by state."""
        return [t for t in self._trials if t.state == state]
    
    def get_values(self) -> List[float]:
        """Get all objective values from completed trials."""
        return [
            t.value for t in self._trials 
            if t.state == TrialState.COMPLETED and t.value is not None
        ]
    
    def get_param_values(self, param_name: str) -> List[Any]:
        """Get all values for a specific parameter."""
        return [
            t.params.get(param_name) for t in self._trials
            if param_name in t.params
        ]
    
    def summary(self) -> StudySummary:
        """Generate study summary."""
        best = self.best_trial
        
        duration = None
        if self._trials:
            start_time = min(t.duration or 0 for t in self._trials)
            end_time = max(
                (t.duration or 0) + (t.intermediate_values[-1][0] if t.intermediate_values else 0)
                for t in self._trials
            )
            duration = end_time - start_time if end_time > start_time else None
        
        return StudySummary(
            study_name=self.study_name,
            direction=self.direction,
            n_trials=self.n_trials,
            n_completed_trials=self.n_completed,
            n_failed_trials=self.n_failed,
            best_value=self.best_value,
            best_params=self.best_params,
            best_model_path=self._best_model_path,
            duration_seconds=duration,
            created_at=self._created_at.isoformat(),
            updated_at=self._updated_at.isoformat(),
        )
    
    def param_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance using fANOVA-style analysis.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if len(self._trials) < 10:
            return {}
        
        # Simple variance-based importance
        completed = [
            t for t in self._trials 
            if t.state == TrialState.COMPLETED and t.value is not None
        ]
        
        if not completed:
            return {}
        
        # Get all parameter names
        param_names = set()
        for trial in completed:
            param_names.update(trial.params.keys())
        
        importance = {}
        values = [t.value for t in completed]
        total_variance = self._variance(values)
        
        if total_variance == 0:
            return {name: 0.0 for name in param_names}
        
        for param in param_names:
            # Calculate conditional variance
            param_values = self.get_param_values(param)
            unique_values = set(v for v in param_values if v is not None)
            
            if len(unique_values) < 2:
                importance[param] = 0.0
                continue
            
            # Group by parameter value and calculate variance reduction
            groups: Dict[Any, List[float]] = {}
            for trial in completed:
                val = trial.params.get(param)
                if val is not None:
                    groups.setdefault(val, []).append(trial.value)
            
            weighted_variance = sum(
                len(g) / len(completed) * self._variance(g)
                for g in groups.values()
                if len(g) > 1
            )
            
            importance[param] = 1 - (weighted_variance / total_variance)
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    @staticmethod
    def _variance(values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def get_optimization_history(self) -> List[Tuple[int, float]]:
        """
        Get optimization history (step, best_value).
        
        Returns:
            List of (trial_number, best_value) tuples
        """
        history = []
        current_best = float("inf") if self.direction == "minimize" else float("-inf")
        
        for i, trial in enumerate(self._trials):
            if trial.state == TrialState.COMPLETED and trial.value is not None:
                if self.direction == "minimize":
                    current_best = min(current_best, trial.value)
                else:
                    current_best = max(current_best, trial.value)
                history.append((i + 1, current_best))
        
        return history
    
    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save study to disk."""
        path = Path(path) if path else (self.storage_path or Path(f"{self.study_name}.json"))
        path = path.with_suffix(".json")
        
        data = {
            "study_name": self.study_name,
            "direction": self.direction,
            "best_model_path": self._best_model_path,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "trials": [t.model_dump() for t in self._trials],
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Study":
        """Load study from disk."""
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        study = cls(
            study_name=data["study_name"],
            direction=data["direction"],
        )
        study._created_at = datetime.fromisoformat(data["created_at"])
        study._updated_at = datetime.fromisoformat(data["updated_at"])
        study._best_model_path = data.get("best_model_path")
        
        for trial_data in data["trials"]:
            trial = TrialResult(**trial_data)
            study._trials.append(trial)
        
        return study
    
    def __repr__(self) -> str:
        return f"Study(name={self.study_name}, n_trials={self.n_trials}, best_value={self.best_value})"
