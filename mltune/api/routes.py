"""
REST API for AutoForge.

Provides FastAPI-based REST endpoints for:
- Experiment management
- Optimization control
- Result retrieval
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mltune.optim.study import Study
from mltune.tracker.backend import SQLiteBackend


# Global state
_db: Optional[SQLiteBackend] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _db
    _db = SQLiteBackend("mltune.db")
    yield
    if _db:
        _db.close()


app = FastAPI(
    title="AutoForge API",
    description="API for AutoForge hyperparameter optimization platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static dashboard (if built)
_root_dir = Path(__file__).resolve().parents[2]
_dashboard_dist = _root_dir / "dashboard" / "dist"
if _dashboard_dist.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_dashboard_dist), html=True),
        name="dashboard",
    )


# Request/Response models
class ExperimentCreate(BaseModel):
    """Request model for creating an experiment."""
    name: str
    config: Dict[str, Any]
    tags: List[str] = []


class ExperimentResponse(BaseModel):
    """Response model for experiment."""
    experiment_id: str
    name: str
    status: str
    config: Dict[str, Any]
    best_value: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    best_model_path: Optional[str] = None


class OptimizeRequest(BaseModel):
    """Request model for starting optimization."""
    experiment_id: str
    n_trials: int = 50
    timeout: Optional[int] = None


class MetricLog(BaseModel):
    """Request model for logging a metric."""
    experiment_id: str
    name: str
    value: float
    step: int


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"name": "AutoForge API", "version": "0.1.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Experiment endpoints
@app.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(request: ExperimentCreate):
    """Create a new experiment."""
    import uuid
    from time import time

    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

    data = {
        "name": request.name,
        "status": "created",
        "config": request.config,
        "tags": request.tags,
        "created_at": time(),
    }

    _db.save_experiment(experiment_id, data)

    return ExperimentResponse(
        experiment_id=experiment_id,
        name=request.name,
        status="created",
        config=request.config,
    )


@app.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
):
    """List all experiments."""
    experiments = _db.list_experiments(limit)

    if status:
        experiments = [e for e in experiments if e.get("status") == status]

    results = []
    for e in experiments:
        name = e.get("name", "")
        # Try to read best_model_path from the corresponding Study file
        model_path: Optional[str] = None
        study = _load_study_from_disk(name)
        if study:
            model_path = study.best_model_path
        results.append(
            ExperimentResponse(
                experiment_id=e["experiment_id"],
                name=name,
                status=e.get("status", "unknown"),
                config=e.get("config", {}),
                best_value=e.get("best_metric"),
                best_params=e.get("best_params"),
                best_model_path=model_path,
            )
        )
    return results


@app.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment by ID."""
    experiment = _db.load_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    name = experiment.get("name", "")
    model_path: Optional[str] = None
    study = _load_study_from_disk(name)
    if study:
        model_path = study.best_model_path

    return ExperimentResponse(
        experiment_id=experiment_id,
        name=name,
        status=experiment.get("status", "unknown"),
        config=experiment.get("config", {}),
        best_value=experiment.get("best_metric"),
        best_params=experiment.get("best_params"),
        best_model_path=model_path,
    )


@app.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment."""
    success = _db.delete_experiment(experiment_id)

    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {"status": "deleted", "experiment_id": experiment_id}


# Metrics endpoints
@app.post("/metrics")
async def log_metric(request: MetricLog):
    """Log a metric value."""
    experiment = _db.load_experiment(request.experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    _db.save_metric(
        experiment_id=request.experiment_id,
        metric_name=request.name,
        value=request.value,
        step=request.step,
    )

    return {"status": "logged"}


@app.get("/experiments/{experiment_id}/metrics")
async def get_metrics(
    experiment_id: str,
    name: Optional[str] = Query(None),
):
    """Get metrics for an experiment."""
    experiment = _db.load_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    metrics = _db.load_metrics(experiment_id, name)
    return {"experiment_id": experiment_id, "metrics": metrics}


# Optimization endpoints
@app.post("/optimize")
async def start_optimization(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """Start an optimization run."""
    experiment = _db.load_experiment(request.experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Update status
    experiment["status"] = "running"
    _db.save_experiment(request.experiment_id, experiment)

    # TODO: Add background optimization task
    # background_tasks.add_task(run_optimization, request)

    return {
        "status": "started",
        "experiment_id": request.experiment_id,
        "n_trials": request.n_trials,
    }


def _load_study_from_disk(study_name: str) -> Optional[Study]:
    """Load a study from disk, always reading the latest file version."""
    candidate_paths = [
        Path("studies") / f"{study_name}.json",
        Path(f"{study_name}.json"),
    ]
    for path in candidate_paths:
        try:
            if path.is_file():
                return Study.load(path)
        except Exception:
            continue
    return None


@app.get("/studies/{study_name}")
async def get_study(study_name: str):
    """Get study results (always reads latest file from disk)."""
    study = _load_study_from_disk(study_name)

    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    return {
        "study_name": study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_model_path": study.best_model_path,
        "n_trials": study.n_trials,
        "summary": study.summary().model_dump(),
        "history": study.get_optimization_history(),
    }


@app.get("/studies/{study_name}/importance")
async def get_param_importance(study_name: str):
    """Get parameter importance."""
    study = _load_study_from_disk(study_name)

    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    importance = study.param_importance()

    return {
        "study_name": study_name,
        "importance": importance,
    }


@app.get("/studies/{study_name}/trials")
async def get_study_trials(study_name: str):
    """Get individual trial details for a study (for live monitoring)."""
    study = _load_study_from_disk(study_name)

    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    trials = []
    for t in study.trials:
        trials.append({
            "trial_id": t.trial_id,
            "params": t.params,
            "value": t.value,
            "state": t.state.value if hasattr(t.state, "value") else str(t.state),
            "duration": t.duration,
            "error": t.error,
        })

    return {
        "study_name": study_name,
        "direction": study.direction,
        "n_completed": study.n_completed,
        "n_failed": study.n_failed,
        "n_trials": study.n_trials,
        "best_value": study.best_value,
        "best_model_path": study.best_model_path,
        "trials": trials,
    }


# Utility endpoints
@app.get("/search_spaces")
async def list_search_space_types():
    """List available search space types."""
    return {
        "types": [
            {"name": "int", "description": "Integer parameter"},
            {"name": "float", "description": "Float parameter"},
            {"name": "loguniform", "description": "Log-uniform parameter"},
            {"name": "categorical", "description": "Categorical parameter"},
        ]
    }


@app.get("/optimizers")
async def list_optimizers():
    """List available optimizers."""
    return {
        "optimizers": [
            {"name": "bayesian", "description": "Bayesian optimization with TPE"},
            {"name": "random", "description": "Random search"},
            {"name": "grid", "description": "Grid search"},
            {"name": "agent", "description": "AI agent optimization"},
        ]
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
