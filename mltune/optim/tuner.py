"""
Tuner - High-level interface for hyperparameter optimization.

The Tuner class provides a simple, unified interface for running
hyperparameter optimization with various strategies.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from mltune.core.config import Config
from mltune.optim.base import BaseOptimizer, Trial, TrialState
from mltune.optim.bayesian import BayesianOptimizer
from mltune.optim.grid import GridOptimizer, RandomOptimizer
from mltune.optim.agent import AgentOptimizer
from mltune.optim.study import Study


class Tuner:
    """
    High-level hyperparameter tuning interface.
    
    The Tuner class provides a unified API for running optimization
    experiments with different strategies.
    
    Example:
        ```python
        from mltune import Tuner, Config
        
        # Define objective function
        def objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            
            model = build_model(lr=lr)
            accuracy = train_and_evaluate(model, batch_size)
            return accuracy
        
        # Create tuner
        config = Config.from_yaml("config.yaml")
        tuner = Tuner(config)
        
        # Run optimization
        study = tuner.optimize(objective, n_trials=100)
        
        print(f"Best accuracy: {study.best_value}")
        print(f"Best params: {study.best_params}")
        ```
    """
    
    OPTIMIZER_MAP = {
        "bayesian": BayesianOptimizer,
        "tpe": BayesianOptimizer,  # TPE is default in Optuna
        "random": RandomOptimizer,
        "grid": GridOptimizer,
        "agent": AgentOptimizer,    # LLM-driven hyperparameter search
    }
    
    def __init__(
        self,
        config: Config,
        strategy: Optional[str] = None,
        loggers: Optional[List[Any]] = None,
        callbacks: Optional[List[Callable]] = None,
        verbose: bool = True,
    ):
        """
        Initialize Tuner.
        
        Args:
            config: Configuration object
            strategy: Optimization strategy (overrides config)
            loggers: List of logger instances
            callbacks: List of callback functions
            verbose: Whether to print progress
        """
        self.config = config
        self.strategy = strategy or config.tuning.strategy
        self.loggers = loggers or []
        self.callbacks = callbacks or []
        self.verbose = verbose
        
        self._optimizer: Optional[BaseOptimizer] = None
        self._study: Optional[Study] = None
        self._console = Console() if verbose else None
    
    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer based on strategy."""
        optimizer_cls = self.OPTIMIZER_MAP.get(self.strategy)
        
        if optimizer_cls is None:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Available: {list(self.OPTIMIZER_MAP.keys())}"
            )
        
        return optimizer_cls(self.config)
    
    def optimize(
        self,
        objective: Optional[Callable[[Trial], float]] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> Study:
        """
        Run hyperparameter optimization.
        
        Args:
            objective: Objective function to optimize
            n_trials: Number of trials (overrides config)
            timeout: Timeout in seconds (overrides config)
            n_jobs: Number of parallel jobs
            show_progress: Whether to show progress bar
            
        Returns:
            Study object with optimization results
        """
        # Create optimizer
        self._optimizer = self._create_optimizer()
        
        # Get parameters from config
        n_trials = n_trials or self.config.tuning.n_trials
        timeout = timeout or self.config.tuning.timeout
        
        # Print header
        if self.verbose and self._console:
            self._print_header(n_trials, timeout)
        
        # Run optimization with progress tracking
        if show_progress and self.verbose:
            study = self._optimize_with_progress(
                objective, n_trials, timeout, n_jobs
            )
        else:
            study = self._optimizer.optimize(
                objective=objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
            )
        
        self._study = study
        
        # Print results
        if self.verbose and self._console:
            self._print_results(study)
        
        # Save study
        if self.config.logging.log_dir:
            save_path = Path(self.config.logging.log_dir) / f"{self.config.experiment.name}_study.json"
            study.save(save_path)
        
        # Log to external loggers
        for logger in self.loggers:
            self._log_to_logger(logger, study)
        
        # Run callbacks
        for callback in self.callbacks:
            callback(study)
        
        return study
    
    def _optimize_with_progress(
        self,
        objective: Callable[[Trial], float],
        n_trials: int,
        timeout: Optional[int],
        n_jobs: int,
    ) -> Study:
        """Run optimization with progress bar."""
        study = Study(
            config=self.config,
            direction=self.config.experiment.direction,
            study_name=self.config.experiment.name,
        )
        
        start_time = time.time()
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• {task.fields[best_value]}"),
            TimeElapsedColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Optimizing...",
                total=n_trials,
                best_value="N/A",
            )
            
            # Incremental save path — so the Dashboard can poll live progress
            _live_save_dir = Path("studies")
            _live_save_dir.mkdir(exist_ok=True)
            _live_save_path = _live_save_dir / f"{study.study_name}.json"

            for i in range(n_trials):
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    break
                
                # Create and run trial
                trial = self._optimizer.create_trial()
                
                try:
                    # Get parameter suggestions
                    params = self._optimizer.suggest(trial)
                    trial.params = params
                    
                    # Run objective
                    value = objective(trial)
                    
                    # Report result
                    result = trial.complete(value)
                    self._optimizer.tell(trial, value)
                    
                    study.add_trial(result)
                    
                    # Update progress
                    best = study.best_value
                    best_str = f"{best:.6f}" if best is not None else "N/A"
                    progress.update(task, advance=1, best_value=best_str)
                    
                except Exception as e:
                    result = trial.fail(str(e))
                    study.add_trial(result)
                    progress.update(task, advance=1)

                # Save after every trial so Dashboard can poll live
                try:
                    study.save(_live_save_path)
                except Exception:
                    pass  # don't break optimization if save fails
        
        return study
    
    def _print_header(self, n_trials: int, timeout: Optional[int]) -> None:
        """Print optimization header."""
        self._console.print("\n" + "=" * 60)
        self._console.print(f"[bold]AutoForge Hyperparameter Optimization[/bold]")
        self._console.print("=" * 60)
        self._console.print(f"  Experiment: {self.config.experiment.name}")
        self._console.print(f"  Strategy: {self.strategy}")
        self._console.print(f"  Direction: {self.config.experiment.direction}")
        self._console.print(f"  Trials: {n_trials}")
        if timeout:
            self._console.print(f"  Timeout: {timeout}s")
        self._console.print("=" * 60 + "\n")
    
    def _print_results(self, study: Study) -> None:
        """Print optimization results."""
        self._console.print("\n" + "=" * 60)
        self._console.print("[bold green]Optimization Complete[/bold green]")
        self._console.print("=" * 60)
        self._console.print(f"  Trials completed: {study.n_completed}/{study.n_trials}")
        self._console.print(f"  Trials failed: {study.n_failed}")
        
        if study.best_value is not None:
            self._console.print(f"\n[bold]Best Value:[/bold] {study.best_value:.6f}")
        
        if study.best_params:
            self._console.print("\n[bold]Best Parameters:[/bold]")
            for key, value in study.best_params.items():
                self._console.print(f"  {key}: {value}")
        
        self._console.print("\n" + "=" * 60 + "\n")
    
    def _log_to_logger(self, logger: Any, study: Study) -> None:
        """Log results to external logger."""
        try:
            if hasattr(logger, "log_study"):
                logger.log_study(study)
            elif hasattr(logger, "log_params") and study.best_params:
                logger.log_params(study.best_params)
                if study.best_value is not None:
                    logger.log_metric("best_value", study.best_value)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to log to {logger}: {e}")
    
    def get_study(self) -> Optional[Study]:
        """Get the current study."""
        return self._study
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found."""
        if self._study:
            return self._study.best_params
        return None
    
    def get_best_value(self) -> Optional[float]:
        """Get the best objective value found."""
        if self._study:
            return self._study.best_value
        return None
    
    @staticmethod
    def quick_optimize(
        objective: Callable[[Trial], float],
        n_trials: int = 50,
        direction: str = "minimize",
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Study:
        """
        Quick optimization without explicit config.
        
        Useful for quick experiments without setting up configuration files.
        
        Example:
            ```python
            study = Tuner.quick_optimize(
                objective=lambda t: train(lr=t.suggest_float("lr", 1e-5, 1e-1, log=True)),
                n_trials=50,
                direction="maximize",
            )
            ```
        """
        config = Config.from_dict({
            "experiment": {
                "name": "quick_optimize",
                "direction": direction,
            },
            "tuning": {
                "strategy": "bayesian",
                "n_trials": n_trials,
                "search_space": search_space or {},
            },
        })
        
        tuner = Tuner(config, verbose=False)
        return tuner.optimize(objective, n_trials=n_trials)
