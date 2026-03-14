"""
Tracker backend implementations.

Provides storage backends for experiment data.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TrackerBackend(ABC):
    """Abstract base class for tracker backends."""

    @abstractmethod
    def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        """Save experiment data."""
        pass

    @abstractmethod
    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment data."""
        pass

    @abstractmethod
    def list_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all experiments."""
        pass

    @abstractmethod
    def save_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """Save a metric value."""
        pass

    @abstractmethod
    def load_metrics(
        self,
        experiment_id: str,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load metrics for an experiment."""
        pass

    @abstractmethod
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        pass


class SQLiteBackend(TrackerBackend):
    """
    SQLite backend for experiment tracking.

    Provides a simple, file-based storage solution suitable for
    single-machine deployments.

    Example:
        ```python
        backend = SQLiteBackend("experiments.db")

        # Save experiment
        backend.save_experiment("exp_001", {
            "name": "My Experiment",
            "config": {"lr": 0.001},
        })

        # Log metrics
        for step in range(100):
            backend.save_metric("exp_001", "loss", 1.0 / (step + 1), step)

        # Load metrics
        metrics = backend.load_metrics("exp_001", "loss")
        ```
    """

    def __init__(self, db_path: Union[str, Path] = "mltune.db"):
        """Initialize SQLite backend."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self.conn

        # Experiments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'created',
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT,
                best_metric REAL,
                best_step INTEGER
            )
        """)

        # Metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                step INTEGER NOT NULL,
                timestamp REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Artifacts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                type TEXT,
                size_bytes INTEGER,
                timestamp REAL,
                metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Create indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_exp "
            "ON metrics(experiment_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_name "
            "ON metrics(metric_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifacts_exp "
            "ON artifacts(experiment_id)"
        )

        conn.commit()

    def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        """Save or update experiment data."""
        conn = self.conn

        # Check if experiment exists
        cursor = conn.execute(
            "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        exists = cursor.fetchone() is not None

        if exists:
            conn.execute(
                """
                UPDATE experiments SET
                    name = ?,
                    status = ?,
                    config = ?,
                    updated_at = CURRENT_TIMESTAMP,
                    tags = ?,
                    best_metric = ?,
                    best_step = ?
                WHERE experiment_id = ?
                """,
                (
                    data.get("name"),
                    data.get("status", "running"),
                    json.dumps(data.get("config", {})),
                    json.dumps(data.get("tags", [])),
                    data.get("best_metric"),
                    data.get("best_step"),
                    experiment_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, status, config, tags, best_metric, best_step
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    data.get("name"),
                    data.get("status", "created"),
                    json.dumps(data.get("config", {})),
                    json.dumps(data.get("tags", [])),
                    data.get("best_metric"),
                    data.get("best_step"),
                ),
            )

        conn.commit()

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment data."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "experiment_id": row["experiment_id"],
            "name": row["name"],
            "status": row["status"],
            "config": json.loads(row["config"]) if row["config"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "best_metric": row["best_metric"],
            "best_step": row["best_step"],
        }

    def list_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all experiments."""
        cursor = self.conn.execute(
            """
            SELECT * FROM experiments
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        experiments = []
        for row in cursor.fetchall():
            experiments.append({
                "experiment_id": row["experiment_id"],
                "name": row["name"],
                "status": row["status"],
                "config": json.loads(row["config"]) if row["config"] else {},
                "created_at": row["created_at"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "best_metric": row["best_metric"],
            })

        return experiments

    def save_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """Save a metric value."""
        import time
        timestamp = timestamp or time.time()

        self.conn.execute(
            """
            INSERT INTO metrics (experiment_id, metric_name, value, step, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (experiment_id, metric_name, value, step, timestamp),
        )
        self.conn.commit()

    def load_metrics(
        self,
        experiment_id: str,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load metrics for an experiment."""
        if metric_name:
            cursor = self.conn.execute(
                """
                SELECT * FROM metrics
                WHERE experiment_id = ? AND metric_name = ?
                ORDER BY step
                """,
                (experiment_id, metric_name),
            )
        else:
            cursor = self.conn.execute(
                """
                SELECT * FROM metrics
                WHERE experiment_id = ?
                ORDER BY step
                """,
                (experiment_id,),
            )

        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                "metric_name": row["metric_name"],
                "value": row["value"],
                "step": row["step"],
                "timestamp": row["timestamp"],
            })

        return metrics

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its data."""
        # Delete metrics
        self.conn.execute(
            "DELETE FROM metrics WHERE experiment_id = ?",
            (experiment_id,),
        )

        # Delete artifacts
        self.conn.execute(
            "DELETE FROM artifacts WHERE experiment_id = ?",
            (experiment_id,),
        )

        # Delete experiment
        cursor = self.conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )

        self.conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class JSONBackend(TrackerBackend):
    """
    JSON file-based backend for experiment tracking.

    Stores each experiment as a separate JSON file, suitable for
    simple deployments and easy debugging.
    """

    def __init__(self, storage_dir: Union[str, Path] = "experiments"):
        """Initialize JSON backend."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_experiment_path(self, experiment_id: str) -> Path:
        """Get path to experiment file."""
        return self.storage_dir / f"{experiment_id}.json"

    def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        """Save experiment data."""
        path = self._get_experiment_path(experiment_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment data."""
        path = self._get_experiment_path(experiment_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all experiments."""
        experiments = []
        for path in sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]:
            with open(path, "r", encoding="utf-8") as f:
                experiments.append(json.load(f))
        return experiments

    def save_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """Save a metric value."""
        import time
        data = self.load_experiment(experiment_id) or {}
        metrics = data.setdefault("metrics", [])
        metrics.append({
            "name": metric_name,
            "value": value,
            "step": step,
            "timestamp": timestamp or time.time(),
        })
        self.save_experiment(experiment_id, data)

    def load_metrics(
        self,
        experiment_id: str,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load metrics for an experiment."""
        data = self.load_experiment(experiment_id)
        if not data:
            return []

        metrics = data.get("metrics", [])
        if metric_name:
            metrics = [m for m in metrics if m["name"] == metric_name]

        return metrics

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        path = self._get_experiment_path(experiment_id)
        if path.exists():
            path.unlink()
            return True
        return False
