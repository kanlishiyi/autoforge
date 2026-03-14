"""
Metrics tracking utilities.

Provides tools for collecting, aggregating, and analyzing metrics.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class MetricValue:
    """A single metric value with metadata."""

    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    """
    Metrics collection and aggregation.

    Example:
        ```python
        tracker = MetricsTracker()

        # Log metrics
        tracker.log("train_loss", 0.5, step=1)
        tracker.log("train_loss", 0.3, step=2)
        tracker.log("val_loss", 0.4, step=1)

        # Get aggregated metrics
        tracker.get_mean("train_loss")  # 0.4
        tracker.get_last("val_loss")   # 0.4

        # Get history
        history = tracker.get_history("train_loss")
        ```
    """

    def __init__(
        self,
        window_size: int = 100,
        aggregators: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of rolling window for statistics
            aggregators: Custom aggregation functions
        """
        self.window_size = window_size
        self._metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._custom_aggregators = aggregators or {}

    def log(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        **context: Any,
    ) -> MetricValue:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Step number (auto-incremented if None)
            **context: Additional context

        Returns:
            MetricValue object
        """
        if step is None:
            step = len(self._metrics[name])

        metric = MetricValue(
            name=name,
            value=value,
            step=step,
            context=context,
        )

        self._metrics[name].append(metric)
        return metric

    def log_dict(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> List[MetricValue]:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Step number
            prefix: Optional prefix for metric names

        Returns:
            List of MetricValue objects
        """
        values = []
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            values.append(self.log(full_name, value, step))
        return values

    def get_history(self, name: str) -> List[MetricValue]:
        """Get full history for a metric."""
        return self._metrics.get(name, [])

    def get_values(self, name: str) -> List[float]:
        """Get values only for a metric."""
        return [m.value for m in self._metrics.get(name, [])]

    def get_last(self, name: str) -> Optional[float]:
        """Get last value for a metric."""
        history = self._metrics.get(name)
        return history[-1].value if history else None

    def get_best(self, name: str, mode: str = "min") -> Optional[float]:
        """
        Get best value for a metric.

        Args:
            name: Metric name
            mode: "min" or "max"

        Returns:
            Best value or None
        """
        values = self.get_values(name)
        if not values:
            return None

        return min(values) if mode == "min" else max(values)

    def get_mean(self, name: str, last_n: Optional[int] = None) -> float:
        """
        Get mean value for a metric.

        Args:
            name: Metric name
            last_n: Only consider last N values (window_size if None)

        Returns:
            Mean value
        """
        values = self.get_values(name)
        if not values:
            return 0.0

        n = last_n or self.window_size
        values = values[-n:]
        return sum(values) / len(values)

    def get_std(self, name: str, last_n: Optional[int] = None) -> float:
        """Get standard deviation for a metric."""
        import math

        values = self.get_values(name)
        if len(values) < 2:
            return 0.0

        n = last_n or self.window_size
        values = values[-n:]

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get all statistics for a metric."""
        values = self.get_values(name)
        if not values:
            return {}

        return {
            "count": len(values),
            "mean": self.get_mean(name),
            "std": self.get_std(name),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
            "best_min": self.get_best(name, "min"),
            "best_max": self.get_best(name, "max"),
        }

    def get_all_names(self) -> List[str]:
        """Get all metric names."""
        return list(self._metrics.keys())

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset metrics.

        Args:
            name: Reset specific metric (all if None)
        """
        if name:
            self._metrics[name] = []
        else:
            self._metrics.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics to dictionary."""
        return {
            name: {
                "values": [m.value for m in history],
                "steps": [m.step for m in history],
                "timestamps": [m.timestamp for m in history],
            }
            for name, history in self._metrics.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsTracker":
        """Create tracker from dictionary."""
        tracker = cls()
        for name, values in data.items():
            for i, value in enumerate(values["values"]):
                tracker._metrics[name].append(
                    MetricValue(
                        name=name,
                        value=value,
                        step=values["steps"][i],
                        timestamp=values["timestamps"][i],
                    )
                )
        return tracker
