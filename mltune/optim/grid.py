"""
Grid search and random search optimizers.

These are baseline optimizers for comparison and for small search spaces.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mltune.core.config import Config, SearchSpaceParam
from mltune.optim.base import BaseOptimizer, Trial, TrialResult


class GridOptimizer(BaseOptimizer):
    """
    Grid search optimizer.
    
    Exhaustively searches all combinations in the search space.
    Best for small, discrete search spaces.
    
    Example:
        ```python
        config = Config.from_yaml("config.yaml")
        optimizer = GridOptimizer(config)
        
        # Define grid
        search_space = {
            "lr": [1e-4, 1e-3, 1e-2],
            "batch_size": [16, 32, 64],
            "layers": [2, 4, 8],
        }
        
        study = optimizer.optimize(objective, search_space=search_space)
        ```
    """
    
    def __init__(self, config: Config, search_space: Optional[Dict[str, List[Any]]] = None):
        super().__init__(config)
        self._grid_space = search_space or {}
        self._grid_points: List[Dict[str, Any]] = []
        self._current_idx = 0
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate all grid points."""
        if not self._grid_space:
            return []
        
        keys = list(self._grid_space.keys())
        values = [self._grid_space[k] for k in keys]
        
        grid = []
        for combo in itertools.product(*values):
            grid.append(dict(zip(keys, combo)))
        
        return grid
    
    def suggest(self, trial: Trial) -> Dict[str, Any]:
        """Get next grid point."""
        if not self._grid_points:
            self._grid_points = self._generate_grid()
        
        if self._current_idx >= len(self._grid_points):
            raise StopIteration("Grid search exhausted")
        
        params = self._grid_points[self._current_idx]
        self._current_idx += 1
        
        return params
    
    def tell(self, trial: Trial, value: float) -> None:
        """Record trial result (no learning in grid search)."""
        pass
    
    def optimize(
        self,
        objective: Optional[Callable[[Trial], float]] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
    ) -> "Study":
        """Run grid search."""
        from mltune.optim.study import Study
        
        objective = objective or self.objective
        if objective is None:
            raise ValueError("No objective function provided")
        
        self._grid_points = self._generate_grid()
        n_trials = min(n_trials or len(self._grid_points), len(self._grid_points))
        
        study = Study(
            config=self.config,
            direction=self.config.experiment.direction,
        )
        
        for i in range(n_trials):
            trial = self.create_trial()
            
            try:
                params = self.suggest(trial)
                trial.params = params
                value = objective(trial)
                result = trial.complete(value)
                study.add_trial(result)
            except StopIteration:
                break
            except Exception as e:
                result = trial.fail(str(e))
                study.add_trial(result)
        
        return study


class RandomOptimizer(BaseOptimizer):
    """
    Random search optimizer.
    
    Samples parameter combinations uniformly at random.
    Often surprisingly effective compared to grid search.
    
    Example:
        ```python
        config = Config.from_yaml("config.yaml")
        optimizer = RandomOptimizer(config)
        
        # Define search space
        search_space = {
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
        }
        
        study = optimizer.optimize(objective, n_trials=100)
        ```
    """
    
    def __init__(
        self,
        config: Config,
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(config)
        self._search_space = search_space or {}
        self._rng = np.random.default_rng(seed)
    
    def suggest(self, trial: Trial) -> Dict[str, Any]:
        """Sample random parameters."""
        params = {}
        
        for name, spec in self._search_space.items():
            params[name] = self._sample_param(spec)
        
        # Also sample from config-defined search space
        for name, param in self.search_space.items():
            if name not in params:
                params[name] = self._sample_from_config(param)
        
        return params
    
    def _sample_param(self, spec: Dict[str, Any]) -> Any:
        """Sample a single parameter based on specification."""
        param_type = spec.get("type", "float")
        
        if param_type == "int":
            low = int(spec.get("low", 0))
            high = int(spec.get("high", 100))
            step = int(spec.get("step", 1))
            values = list(range(low, high + 1, step))
            return self._rng.choice(values)
        
        elif param_type == "float":
            low = spec.get("low", 0.0)
            high = spec.get("high", 1.0)
            return self._rng.uniform(low, high)
        
        elif param_type == "loguniform":
            low = spec.get("low", 1e-6)
            high = spec.get("high", 1.0)
            log_low = np.log(low)
            log_high = np.log(high)
            return np.exp(self._rng.uniform(log_low, log_high))
        
        elif param_type == "categorical":
            choices = spec.get("choices", [])
            return self._rng.choice(choices) if choices else None
        
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _sample_from_config(self, param: SearchSpaceParam) -> Any:
        """Sample from config-defined search space."""
        if param.type == "int":
            return self._rng.integers(
                int(param.low),
                int(param.high) + 1,
                step=int(param.step) if param.step else 1,
            )
        elif param.type == "float":
            return self._rng.uniform(param.low, param.high)
        elif param.type == "loguniform":
            log_low = np.log(param.low)
            log_high = np.log(param.high)
            return np.exp(self._rng.uniform(log_low, log_high))
        elif param.type == "categorical":
            return self._rng.choice(param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {param.type}")
    
    def tell(self, trial: Trial, value: float) -> None:
        """Record trial result (no learning in random search)."""
        pass
    
    def optimize(
        self,
        objective: Optional[Callable[[Trial], float]] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
    ) -> "Study":
        """Run random search."""
        from mltune.optim.study import Study
        
        objective = objective or self.objective
        if objective is None:
            raise ValueError("No objective function provided")
        
        n_trials = n_trials or self.config.tuning.n_trials
        
        study = Study(
            config=self.config,
            direction=self.config.experiment.direction,
        )
        
        for i in range(n_trials):
            trial = self.create_trial()
            
            try:
                params = self.suggest(trial)
                trial.params = params
                value = objective(trial)
                result = trial.complete(value)
                study.add_trial(result)
            except Exception as e:
                result = trial.fail(str(e))
                study.add_trial(result)
        
        return study


class RandomTrial(Trial):
    """Trial with suggest methods for random sampling."""
    
    def __init__(self, trial_id: int, rng: np.random.Generator):
        super().__init__(trial_id)
        self._rng = rng
    
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: Optional[float] = None,
    ) -> float:
        """Suggest a float parameter."""
        if name in self.params:
            return self.params[name]
        
        if log:
            value = np.exp(self._rng.uniform(np.log(low), np.log(high)))
        elif step:
            n_steps = int((high - low) / step)
            value = low + self._rng.integers(0, n_steps + 1) * step
        else:
            value = self._rng.uniform(low, high)
        
        self.params[name] = value
        return value
    
    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        """Suggest an integer parameter."""
        if name in self.params:
            return self.params[name]
        
        if log:
            value = int(np.exp(self._rng.uniform(np.log(low), np.log(high))))
            value = max(low, min(high, value))
        else:
            n_steps = (high - low) // step
            value = low + self._rng.integers(0, n_steps + 1) * step
        
        self.params[name] = value
        return value
    
    def suggest_categorical(
        self,
        name: str,
        choices: List[Any],
    ) -> Any:
        """Suggest a categorical parameter."""
        if name in self.params:
            return self.params[name]
        
        value = self._rng.choice(choices)
        self.params[name] = value
        return value
