"""
Bayesian optimizer using Optuna TPE sampler.

This is the primary optimizer for AutoForge, providing efficient
hyperparameter optimization using Tree-structured Parzen Estimator.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from mltune.core.config import Config, SearchSpaceParam
from mltune.optim.base import BaseOptimizer, Trial, TrialResult, TrialState


class OptunaTrial(Trial):
    """Trial wrapper for Optuna."""
    
    def __init__(self, trial_id: int, optuna_trial: Any):
        super().__init__(trial_id)
        self._optuna_trial = optuna_trial
    
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
        return self._optuna_trial.suggest_float(
            name, low, high, log=log, step=step
        )
    
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
        return self._optuna_trial.suggest_int(
            name, low, high, step=step, log=log
        )
    
    def suggest_categorical(
        self,
        name: str,
        choices: List[Any],
    ) -> Any:
        """Suggest a categorical parameter."""
        return self._optuna_trial.suggest_categorical(name, choices)
    
    def should_prune(self) -> bool:
        """Check if trial should be pruned."""
        try:
            import optuna
            self._optuna_trial.report(self._intermediate_values[-1][1], self._step)
            return self._optuna_trial.should_prune()
        except Exception:
            return False


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization using Optuna's TPE sampler.
    
    This optimizer provides efficient hyperparameter search using
    Tree-structured Parzen Estimator (TPE), which models the
    objective function and focuses sampling on promising regions.
    
    Features:
    - Efficient exploration of parameter space
    - Support for conditional parameters
    - Pruning of unpromising trials
    - Multi-objective optimization support
    
    Example:
        ```python
        config = Config.from_yaml("config.yaml")
        optimizer = BayesianOptimizer(config)
        
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            return train_and_evaluate(lr)
        
        study = optimizer.optimize(objective, n_trials=100)
        ```
    """
    
    def __init__(
        self,
        config: Config,
        sampler: Optional[Any] = None,
        pruner: Optional[Any] = None,
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            config: Configuration object
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)
        """
        super().__init__(config)
        
        # Import Optuna
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install with: pip install optuna"
            )
        
        self._optuna = optuna
        
        # Set up sampler and pruner
        self._sampler = sampler or TPESampler(seed=config.tuning.seed)
        self._pruner = pruner if pruner is not None else MedianPruner()
        
        # Create Optuna study
        self._study = optuna.create_study(
            direction=config.experiment.direction,
            sampler=self._sampler,
            pruner=self._pruner,
            study_name=config.experiment.name,
        )
    
    def suggest(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial.
        
        For Bayesian optimization, parameter suggestion happens
        inside the objective function via trial.suggest_* methods.
        """
        # With Optuna, we use trial.suggest_* methods inside objective
        return {}
    
    def tell(self, trial: Trial, value: float) -> None:
        """
        Report the result of a trial.
        
        With Optuna, this is handled internally.
        """
        pass
    
    def create_trial(self) -> Trial:
        """Create a new trial."""
        self._trial_id_counter += 1
        return Trial(trial_id=self._trial_id_counter - 1)
    
    def optimize(
        self,
        objective: Optional[Callable[[Trial], float]] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
    ) -> "Study":
        """
        Run Bayesian optimization.
        
        Args:
            objective: Objective function
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            
        Returns:
            Study with optimization results
        """
        objective = objective or self.objective
        if objective is None:
            raise ValueError("No objective function provided")
        
        n_trials = n_trials or self.config.tuning.n_trials
        timeout = timeout or self.config.tuning.timeout
        
        # Wrap objective to handle AutoForge Trial
        def optuna_objective(optuna_trial):
            # Create wrapper trial
            trial = OptunaTrial(self._trial_id_counter, optuna_trial)
            self._trial_id_counter += 1
            
            # Add search space suggestions if defined in config
            for param_name, param_def in self.search_space.items():
                if param_name not in trial.params:
                    value = self._sample_from_config(
                        param_name, param_def, optuna_trial
                    )
                    trial.params[param_name] = value
            
            # Run objective
            try:
                value = objective(trial)
                trial.complete(value)
                return value
            except Exception as e:
                trial.fail(str(e))
                raise
        
        # Run Optuna optimization
        self._study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=(Exception,),
        )
        
        # Convert to AutoForge Study
        return self._convert_study()
    
    def _sample_from_config(
        self,
        name: str,
        param: SearchSpaceParam,
        trial: Any,
    ) -> Any:
        """Sample parameter based on config definition."""
        if param.type == "int":
            return trial.suggest_int(
                name,
                int(param.low),
                int(param.high),
                step=int(param.step) if param.step else 1,
                log=param.log,
            )
        elif param.type == "float":
            return trial.suggest_float(
                name,
                param.low,
                param.high,
                log=param.log,
                step=param.step,
            )
        elif param.type == "loguniform":
            return trial.suggest_float(name, param.low, param.high, log=True)
        elif param.type == "categorical":
            return trial.suggest_categorical(name, param.choices)
        else:
            raise ValueError(f"Unknown parameter type: {param.type}")
    
    def _convert_study(self) -> "Study":
        """Convert Optuna study to AutoForge Study."""
        from mltune.optim.study import Study
        
        study = Study(
            config=self.config,
            direction=self.config.experiment.direction,
            study_name=self.config.experiment.name,
        )
        
        for i, trial in enumerate(self._study.trials):
            result = TrialResult(
                trial_id=i,
                params=dict(trial.params),
                value=trial.value,
                state=self._convert_state(trial.state),
                duration=trial.duration,
                error=str(trial.user_attrs.get("error")) if trial.value is None else None,
                intermediate_values=[(s, v) for s, v in trial.intermediate_values.items()],
                user_attrs=dict(trial.user_attrs),
            )
            study.add_trial(result)
        
        return study
    
    @staticmethod
    def _convert_state(optuna_state: Any) -> TrialState:
        """Convert Optuna trial state to AutoForge state."""
        import optuna
        
        state_map = {
            optuna.trial.TrialState.COMPLETE: TrialState.COMPLETED,
            optuna.trial.TrialState.FAIL: TrialState.FAILED,
            optuna.trial.TrialState.PRUNED: TrialState.PRUNED,
            optuna.trial.TrialState.RUNNING: TrialState.RUNNING,
        }
        return state_map.get(optuna_state, TrialState.RUNNING)
    
    def get_param_importance(self) -> Dict[str, float]:
        """Get parameter importance from Optuna."""
        try:
            importance = self._optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception:
            return {}
