"""
Base optimizer interface for AutoForge.

Defines the abstract interface that all optimizers must implement.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
import random

from pydantic import BaseModel

from mltune.core.config import Config, SearchSpaceParam


class TrialState(str, Enum):
    """State of an optimization trial."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


class TrialResult(BaseModel):
    """Result of a single optimization trial."""
    
    trial_id: int
    params: Dict[str, Any]
    value: Optional[float] = None
    state: TrialState = TrialState.RUNNING
    duration: Optional[float] = None
    error: Optional[str] = None
    intermediate_values: List[Tuple[int, float]] = []
    
    # Additional metadata
    user_attrs: Dict[str, Any] = {}
    system_attrs: Dict[str, Any] = {}


@dataclass
class Trial:
    """
    Represents a single optimization trial.
    
    Trials are created by the optimizer and passed to the objective function.
    The objective function uses the trial to:
    1. Sample hyperparameters from the search space
    2. Report intermediate results
    3. Report the final result
    """
    
    trial_id: int
    params: Dict[str, Any] = field(default_factory=dict)
    state: TrialState = TrialState.RUNNING
    result: Optional[TrialResult] = None
    
    # Internal tracking
    _intermediate_values: List[Tuple[int, float]] = field(default_factory=list)
    _user_attrs: Dict[str, Any] = field(default_factory=dict)
    _step: int = 0
    _start_time: float = field(default_factory=time.time)
    
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: Optional[float] = None,
    ) -> float:
        """Suggest a float parameter using simple random sampling."""
        if low >= high:
            return float(low)
        if log:
            # Sample uniformly in log space
            log_low = math.log(low)
            log_high = math.log(high)
            value = math.exp(random.uniform(log_low, log_high))
        else:
            value = random.uniform(low, high)
        if step is not None and step > 0:
            # Quantize to the nearest multiple of step within [low, high]
            k = round((value - low) / step)
            value = low + k * step
            value = max(min(value, high), low)
        # Store in params for bookkeeping
        self.params[name] = value
        return float(value)
    
    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        """Suggest an integer parameter using simple random sampling."""
        if low >= high:
            value = int(low)
        else:
            if log and low > 0 and high > 0:
                # Log-uniform over integers: sample in log space then round
                log_low = math.log(low)
                log_high = math.log(high)
                v = math.exp(random.uniform(log_low, log_high))
                value = int(round(v))
            else:
                if step <= 0:
                    step = 1
                # Sample from the discrete set {low, low+step, ..., high}
                n_steps = max((high - low) // step, 0)
                idx = random.randint(0, n_steps)
                value = low + idx * step
        value = max(min(value, high), low)
        self.params[name] = value
        return int(value)
    
    def suggest_categorical(
        self,
        name: str,
        choices: List[Any],
    ) -> Any:
        """Suggest a categorical parameter using uniform random choice."""
        if not choices:
            raise ValueError("choices must be a non-empty list")
        value = random.choice(choices)
        self.params[name] = value
        return value
    
    def report(self, value: float, step: Optional[int] = None) -> None:
        """
        Report an intermediate value.
        
        Args:
            value: Metric value
            step: Step number (auto-incremented if None)
        """
        if step is None:
            step = self._step
            self._step += 1
        
        self._intermediate_values.append((step, value))
    
    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a user attribute."""
        self._user_attrs[key] = value
    
    def get_user_attr(self, key: str, default: Any = None) -> Any:
        """Get a user attribute."""
        return self._user_attrs.get(key, default)
    
    def should_prune(self) -> bool:
        """Check if trial should be pruned."""
        return False
    
    def complete(self, value: float) -> TrialResult:
        """Mark trial as completed with final value."""
        self.state = TrialState.COMPLETED
        self.result = TrialResult(
            trial_id=self.trial_id,
            params=self.params,
            value=value,
            state=TrialState.COMPLETED,
            duration=time.time() - self._start_time,
            intermediate_values=self._intermediate_values.copy(),
            user_attrs=self._user_attrs.copy(),
        )
        return self.result
    
    def fail(self, error: str) -> TrialResult:
        """Mark trial as failed."""
        self.state = TrialState.FAILED
        self.result = TrialResult(
            trial_id=self.trial_id,
            params=self.params,
            state=TrialState.FAILED,
            duration=time.time() - self._start_time,
            error=error,
            user_attrs=self._user_attrs.copy(),
        )
        return self.result


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    All optimizers must implement the following methods:
    - suggest(): Generate parameters for a trial
    - tell(): Report results for a trial
    - optimize(): Run the full optimization loop
    """
    
    def __init__(
        self,
        config: Config,
        objective: Optional[Callable[[Trial], float]] = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            config: Configuration object
            objective: Objective function to optimize
        """
        self.config = config
        self.objective = objective
        self.search_space = config.tuning.search_space
        self.direction = config.experiment.direction
        self._trials: List[Trial] = []
        self._trial_id_counter = 0
    
    def create_trial(self) -> Trial:
        """Create a new trial."""
        trial = Trial(trial_id=self._trial_id_counter)
        self._trial_id_counter += 1
        self._trials.append(trial)
        return trial
    
    @abstractmethod
    def suggest(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial.
        
        Args:
            trial: Trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        pass
    
    @abstractmethod
    def tell(self, trial: Trial, value: float) -> None:
        """
        Report the result of a trial.
        
        Args:
            trial: Completed trial
            value: Objective value
        """
        pass
    
    def optimize(
        self,
        objective: Optional[Callable[[Trial], float]] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
    ) -> "Study":
        """
        Run the optimization loop.
        
        Args:
            objective: Objective function
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            
        Returns:
            Study object with results
        """
        objective = objective or self.objective
        if objective is None:
            raise ValueError("No objective function provided")
        
        n_trials = n_trials or self.config.tuning.n_trials
        timeout = timeout or self.config.tuning.timeout
        
        start_time = time.time()
        study = Study(
            config=self.config,
            direction=self.direction,
        )

        # Incremental save path — so the Dashboard can poll live progress
        from pathlib import Path as _Path
        _live_save_dir = _Path("studies")
        _live_save_dir.mkdir(exist_ok=True)
        _live_save_path = _live_save_dir / f"{study.study_name}.json"
        
        for i in range(n_trials):
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Create and run trial
            trial = self.create_trial()
            
            try:
                # Get parameter suggestions
                params = self.suggest(trial)
                trial.params = params
                
                # Run objective
                value = objective(trial)
                
                # Report result
                result = trial.complete(value)
                self.tell(trial, value)
                
                study.add_trial(result)
                
            except Exception as e:
                result = trial.fail(str(e))
                study.add_trial(result)

            # Save after every trial so Dashboard can poll live
            try:
                study.save(_live_save_path)
            except Exception:
                pass
        
        return study
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found so far."""
        completed_trials = [
            t for t in self._trials 
            if t.state == TrialState.COMPLETED and t.result is not None
        ]
        
        if not completed_trials:
            return None
        
        if self.direction == "minimize":
            best_trial = min(completed_trials, key=lambda t: t.result.value)
        else:
            best_trial = max(completed_trials, key=lambda t: t.result.value)
        
        return best_trial.params
    
    def get_best_value(self) -> Optional[float]:
        """Get the best objective value found so far."""
        completed_trials = [
            t for t in self._trials 
            if t.state == TrialState.COMPLETED and t.result is not None
        ]
        
        if not completed_trials:
            return None
        
        values = [t.result.value for t in completed_trials]
        
        if self.direction == "minimize":
            return min(values)
        else:
            return max(values)
    
    @staticmethod
    def _sample_param(param: SearchSpaceParam, trial: Trial) -> Any:
        """Sample a parameter value based on its definition."""
        if param.type == "int":
            return trial.suggest_int(
                param.name if hasattr(param, "name") else "param",
                param.low,
                param.high,
                step=param.step or 1,
                log=param.log,
            )
        elif param.type == "float":
            return trial.suggest_float(
                param.name if hasattr(param, "name") else "param",
                param.low,
                param.high,
                log=param.log,
                step=param.step,
            )
        elif param.type == "loguniform":
            return trial.suggest_float(
                param.name if hasattr(param, "name") else "param",
                param.low,
                param.high,
                log=True,
            )
        elif param.type == "categorical":
            return trial.suggest_categorical(
                param.name if hasattr(param, "name") else "param",
                param.choices,
            )
        else:
            raise ValueError(f"Unknown parameter type: {param.type}")
