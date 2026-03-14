"""
Visualization utilities for experiment analysis.

Provides plotting and visualization tools for:
- Learning curves
- Optimization history
- Parameter importance
- Experiment comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json


class Visualizer:
    """
    Visualization generator for experiments.
    
    Creates static visualizations using matplotlib/plotly and
    generates data for interactive dashboards.
    
    Example:
        ```python
        from mltune.tracker import Visualizer
        
        viz = Visualizer()
        
        # Plot learning curve
        viz.plot_learning_curve(
            steps=[1, 2, 3, 4, 5],
            values=[0.9, 0.7, 0.5, 0.3, 0.2],
            name="train_loss",
            save_path="learning_curve.png",
        )
        
        # Plot optimization history
        viz.plot_optimization_history(
            history=[(1, 0.9), (2, 0.7), (3, 0.5)],
            save_path="optim_history.png",
        )
        ```
    """
    
    def __init__(
        self,
        backend: str = "matplotlib",
        style: str = "seaborn-v0_8-whitegrid",
    ):
        """
        Initialize visualizer.
        
        Args:
            backend: Plotting backend ("matplotlib" or "plotly")
            style: Matplotlib style
        """
        self.backend = backend
        self.style = style
        
        if backend == "matplotlib":
            try:
                import matplotlib.pyplot as plt
                plt.style.use(style)
                self._plt = plt
            except ImportError:
                raise ImportError("matplotlib is required: pip install matplotlib")
    
    def plot_learning_curve(
        self,
        steps: List[int],
        values: List[float],
        name: str = "metric",
        title: Optional[str] = None,
        xlabel: str = "Step",
        ylabel: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Plot a learning curve.
        
        Args:
            steps: Step numbers
            values: Metric values
            name: Metric name
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure
            show: Whether to display the plot
            **kwargs: Additional arguments for plot
            
        Returns:
            Figure object
        """
        if self.backend == "matplotlib":
            fig, ax = self._plt.subplots(figsize=(10, 6))
            
            ax.plot(steps, values, label=name, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel or name)
            ax.set_title(title or f"{name} Learning Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            
            if show:
                self._plt.show()
            
            return fig
        
        return None
    
    def plot_optimization_history(
        self,
        history: List[Tuple[int, float]],
        best_values: Optional[List[float]] = None,
        title: str = "Optimization History",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot optimization history.
        
        Args:
            history: List of (trial_number, value) tuples
            best_values: Running best values (computed if None)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        if not history:
            return None
        
        trials, values = zip(*history)
        
        if best_values is None:
            best_values = []
            current_best = float("inf")
            for v in values:
                current_best = min(current_best, v)
                best_values.append(current_best)
        
        if self.backend == "matplotlib":
            fig, ax = self._plt.subplots(figsize=(10, 6))
            
            ax.scatter(trials, values, alpha=0.5, label="Trial values")
            ax.plot(trials, best_values, "r-", linewidth=2, label="Best value")
            
            ax.set_xlabel("Trial")
            ax.set_ylabel("Objective Value")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            
            if show:
                self._plt.show()
            
            return fig
        
        return None
    
    def plot_param_importance(
        self,
        importance: Dict[str, float],
        title: str = "Parameter Importance",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot parameter importance as bar chart.
        
        Args:
            importance: Dictionary of param_name -> importance_score
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        if not importance:
            return None
        
        if self.backend == "matplotlib":
            # Sort by importance
            sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            names, values = zip(*sorted_items)
            
            fig, ax = self._plt.subplots(figsize=(10, max(6, len(names) * 0.5)))
            
            bars = ax.barh(range(len(names)), values)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel("Importance")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="x")
            
            # Color bars
            for bar, val in zip(bars, values):
                bar.set_color(self._plt.cm.Blues(0.3 + 0.7 * val))
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            
            if show:
                self._plt.show()
            
            return fig
        
        return None
    
    def plot_slice(
        self,
        param_values: Dict[str, List[Any]],
        objective_values: List[float],
        param_name: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot slice plot for a single parameter.
        
        Args:
            param_values: Dictionary of parameter values per trial
            objective_values: Objective values per trial
            param_name: Parameter to plot
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        values = param_values.get(param_name, [])
        if not values:
            return None
        
        if self.backend == "matplotlib":
            fig, ax = self._plt.subplots(figsize=(10, 6))
            
            ax.scatter(values, objective_values, alpha=0.5)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Objective Value")
            ax.set_title(title or f"Slice Plot: {param_name}")
            ax.grid(True, alpha=0.3)
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            
            if show:
                self._plt.show()
            
            return fig
        
        return None
    
    def plot_contour(
        self,
        param_values: Dict[str, List[Any]],
        objective_values: List[float],
        param_x: str,
        param_y: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot contour plot for two parameters.
        
        Args:
            param_values: Dictionary of parameter values per trial
            objective_values: Objective values per trial
            param_x: X-axis parameter
            param_y: Y-axis parameter
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        x_values = param_values.get(param_x, [])
        y_values = param_values.get(param_y, [])
        
        if not x_values or not y_values:
            return None
        
        if self.backend == "matplotlib":
            import numpy as np
            from scipy.interpolate import griddata
            
            fig, ax = self._plt.subplots(figsize=(10, 8))
            
            # Create grid
            x = np.array(x_values)
            y = np.array(y_values)
            z = np.array(objective_values)
            
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="cubic")
            
            # Plot contour
            contour = ax.contourf(xi, yi, zi, levels=15, cmap="viridis_r")
            self._plt.colorbar(contour, ax=ax, label="Objective")
            
            # Scatter actual points
            ax.scatter(x, y, c="white", edgecolors="black", s=50, alpha=0.8)
            
            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            ax.set_title(title or f"Contour: {param_x} vs {param_y}")
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            
            if show:
                self._plt.show()
            
            return fig
        
        return None
    
    def plot_comparison(
        self,
        experiments: Dict[str, Dict[str, List[float]]],
        metric_name: str,
        steps: Optional[List[int]] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot comparison of multiple experiments.
        
        Args:
            experiments: Dict of exp_name -> {"values": [...], "steps": [...]}
            metric_name: Metric to compare
            steps: Common steps (derived if None)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        if not experiments:
            return None
        
        if self.backend == "matplotlib":
            fig, ax = self._plt.subplots(figsize=(12, 6))
            
            colors = self._plt.cm.tab10.colors
            
            for i, (name, data) in enumerate(experiments.items()):
                values = data.get("values", [])
                exp_steps = data.get("steps", list(range(len(values))))
                ax.plot(
                    exp_steps, values,
                    label=name,
                    color=colors[i % len(colors)],
                    linewidth=2,
                )
            
            ax.set_xlabel("Step")
            ax.set_ylabel(metric_name)
            ax.set_title(title or f"Comparison: {metric_name}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            
            if show:
                self._plt.show()
            
            return fig
        
        return None
    
    def generate_report_data(
        self,
        study: Any,
        include_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate data for an interactive report.
        
        Args:
            study: Study object
            include_plots: Whether to include plot data
            
        Returns:
            Dictionary with report data
        """
        report = {
            "summary": study.summary().model_dump() if hasattr(study, "summary") else {},
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "params": t.params,
                    "value": t.value,
                    "state": t.state.value if hasattr(t.state, "value") else str(t.state),
                }
                for t in study.trials
            ],
        }
        
        if include_plots:
            # Add optimization history data
            history = study.get_optimization_history()
            report["plots"] = {
                "optimization_history": {
                    "trials": [h[0] for h in history],
                    "values": [h[1] for h in history],
                },
                "param_importance": study.param_importance(),
            }
        
        return report
