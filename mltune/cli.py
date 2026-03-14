"""
Command Line Interface for AutoForge.

Provides commands for:
- Running optimization
- Managing experiments
- Starting API server
- Viewing results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="mltune")
def main():
    """AutoForge - Intelligent Machine Learning Training & Tuning Platform."""
    pass


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n-trials", "-n", default=50, help="Number of trials")
@click.option("--timeout", "-t", default=None, help="Timeout in seconds")
@click.option("--strategy", "-s", default="bayesian", help="Optimization strategy")
@click.option("--output", "-o", default=None, help="Output path for results")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def optimize(
    config_path: str,
    n_trials: int,
    timeout: Optional[int],
    strategy: str,
    output: Optional[str],
    verbose: bool,
):
    """
    Run hyperparameter optimization.
    
    CONFIG_PATH: Path to configuration file (YAML or JSON)
    """
    from mltune import Config, Tuner
    
    # Load configuration
    config_path = Path(config_path)
    console.print(f"[bold blue]Loading configuration from {config_path}[/]")
    
    if config_path.suffix in (".yaml", ".yml"):
        config = Config.from_yaml(config_path)
    else:
        config = Config.from_json(config_path)
    
    # Override settings
    config.tuning.n_trials = n_trials
    config.tuning.strategy = strategy
    if timeout:
        config.tuning.timeout = timeout
    
    # Create tuner
    tuner = Tuner(config, verbose=verbose)
    
    console.print(f"[bold green]Starting optimization[/]")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Trials: {n_trials}")
    console.print(f"  Direction: {config.experiment.direction}")
    
    # Define placeholder objective (user should provide their own)
    def objective(trial):
        # Sample from config search space
        params = {}
        for name, param in config.tuning.search_space.items():
            if param.type == "float":
                params[name] = trial.suggest_float(name, param.low, param.high, log=param.log)
            elif param.type == "int":
                params[name] = trial.suggest_int(name, int(param.low), int(param.high))
            elif param.type == "categorical":
                params[name] = trial.suggest_categorical(name, param.choices)
        
        # Placeholder: user should implement actual training
        console.print(f"[dim]Trial params: {params}[/]")
        return 0.0
    
    # Run optimization
    study = tuner.optimize(objective, n_trials=n_trials)
    
    # Display results
    console.print("\n[bold green]Optimization Complete[/]")
    console.print(f"  Best value: {study.best_value}")
    console.print(f"  Best params: {study.best_params}")
    
    # Save results
    if output:
        output_path = Path(output)
        study.save(output_path)
        console.print(f"\n[bold]Results saved to {output_path}[/]")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--epochs", "-e", default=None, help="Override number of epochs")
@click.option("--output", "-o", default=None, help="Output directory")
def train(config_path: str, epochs: Optional[int], output: Optional[str]):
    """
    Run a single training experiment.
    
    CONFIG_PATH: Path to configuration file
    """
    from mltune import Config, Experiment
    
    # Load configuration
    config_path = Path(config_path)
    config = Config.from_yaml(config_path) if config_path.suffix in (".yaml", ".yml") else Config.from_json(config_path)
    
    if epochs:
        config.training.epochs = epochs
    
    console.print(f"[bold blue]Starting training: {config.experiment.name}[/]")
    
    # Create experiment
    exp = Experiment(
        name=config.experiment.name,
        config=config,
        storage_dir=output or "experiments",
    )
    
    with exp.track():
        # Placeholder training loop
        for epoch in range(config.training.epochs):
            # User should implement actual training
            loss = 1.0 / (epoch + 1)
            exp.log_metric("train_loss", loss, step=epoch)
            console.print(f"Epoch {epoch + 1}/{config.training.epochs}: loss={loss:.4f}")
    
    console.print(f"\n[bold green]Training Complete[/]")
    console.print(f"  Final loss: {exp.get_best_metric('train_loss')}")


@main.command()
@click.option("--limit", "-l", default=20, help="Number of experiments to show")
@click.option("--status", "-s", default=None, help="Filter by status")
def experiments(limit: int, status: Optional[str]):
    """List all experiments."""
    from mltune.tracker.backend import SQLiteBackend
    
    backend = SQLiteBackend()
    experiments = backend.list_experiments(limit)
    
    if status:
        experiments = [e for e in experiments if e.get("status") == status]
    
    if not experiments:
        console.print("[yellow]No experiments found[/]")
        return
    
    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Best Value", style="magenta")
    table.add_column("Created")
    
    for exp in experiments:
        table.add_row(
            exp.get("experiment_id", "N/A"),
            exp.get("name", "N/A"),
            exp.get("status", "N/A"),
            str(exp.get("best_metric", "N/A")),
            exp.get("created_at", "N/A"),
        )
    
    console.print(table)


@main.command()
@click.argument("experiment_id")
def show(experiment_id: str):
    """Show experiment details."""
    from mltune.tracker.backend import SQLiteBackend
    
    backend = SQLiteBackend()
    experiment = backend.load_experiment(experiment_id)
    
    if not experiment:
        console.print(f"[red]Experiment not found: {experiment_id}[/]")
        return
    
    console.print(f"\n[bold blue]Experiment: {experiment_id}[/]")
    console.print(f"  Name: {experiment.get('name')}")
    console.print(f"  Status: {experiment.get('status')}")
    console.print(f"  Created: {experiment.get('created_at')}")
    
    if experiment.get("best_metric"):
        console.print(f"  Best Metric: {experiment.get('best_metric')}")
    
    if experiment.get("config"):
        console.print("\n[bold]Configuration:[/]")
        console.print(json.dumps(experiment.get("config"), indent=2))
    
    # Show metrics
    metrics = backend.load_metrics(experiment_id)
    if metrics:
        console.print(f"\n[bold]Metrics ({len(metrics)} records):[/]")
        # Group by metric name
        metric_names = set(m["metric_name"] for m in metrics)
        for name in metric_names:
            values = [m["value"] for m in metrics if m["metric_name"] == name]
            console.print(f"  {name}: min={min(values):.4f}, max={max(values):.4f}, last={values[-1]:.4f}")


@main.command()
@click.argument("experiment_id")
@click.confirmation_option(prompt="Are you sure you want to delete this experiment?")
def delete(experiment_id: str):
    """Delete an experiment."""
    from mltune.tracker.backend import SQLiteBackend
    
    backend = SQLiteBackend()
    success = backend.delete_experiment(experiment_id)
    
    if success:
        console.print(f"[green]Deleted experiment: {experiment_id}[/]")
    else:
        console.print(f"[red]Failed to delete experiment: {experiment_id}[/]")


@main.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to bind to")
def server(host: str, port: int):
    """Start the API server."""
    from mltune.api.routes import run_server
    
    console.print(f"[bold green]Starting API server on {host}:{port}[/]")
    run_server(host=host, port=port)


@main.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8080, help="Port to bind to")
def dashboard(host: str, port: int):
    """Start the web dashboard (API server + static dashboard)."""
    from mltune.api.routes import run_server
    
    console.print(f"[bold blue]Starting dashboard on http://{host}:{port}/dashboard[/]")
    console.print(
        "[dim]Make sure you have built the frontend in 'dashboard/' "
        "(npm install && npm run build) so that static files are available.[/]"
    )
    run_server(host=host, port=port)


@main.command(name="lgbm-stock")
@click.option("--n-trials", "-n", default=30, help="Number of optimization trials")
@click.option("--study-name", default="lgbm_stock_price", help="Study name / output file stem")
def lgbm_stock_example(n_trials: int, study_name: str):
    """
    Example: Use LightGBM to tune a stock price regression model.

    This is a self-contained example that:
    - creates a Config with a sensible LightGBM search space
    - runs Bayesian optimization over RMSE on a synthetic dataset
    - saves the resulting Study JSON into the `studies/` directory

    NOTE:
        You need extra dependencies installed:
            pip install lightgbm scikit-learn

        Replace the `load_stock_data` function with your own data loader
        (e.g. reading from CSV / database / parquet) for real use cases.
    """
    try:
        import numpy as np  # type: ignore[import]
        import pandas as pd  # type: ignore[import]
        import yfinance as yf  # type: ignore[import]
        import lightgbm as lgb  # type: ignore[import]
        from sklearn.model_selection import train_test_split  # type: ignore[import]
    except ImportError:
        console.print(
            "[red]Missing dependencies.[/]\n"
            "Please install the extra packages first:\n"
            "  pip install lightgbm scikit-learn pandas yfinance"
        )
        raise click.Abort()

    from mltune import Config, Tuner
    from mltune.optim.base import Trial

    console.print("[bold blue]Running LightGBM stock price tuning example[/]")

    # --- 1. Data loading: local CSV cache → yfinance → realistic synthetic ---
    def _build_features(df):
        """Given a DataFrame with 'Close' and 'Volume', add technical indicators and target."""
        df = df[["Close", "Volume"]].copy().dropna()

        # Basic returns and rolling statistics
        df["ret_1"] = df["Close"].pct_change()
        for k in (3, 5, 10):
            df[f"ma_{k}"] = df["Close"].rolling(k).mean()
            df[f"std_{k}"] = df["Close"].rolling(k).std()
            df[f"ret_{k}"] = df["Close"].pct_change(k)

        # Target: next-day return
        df["target"] = df["Close"].shift(-1) / df["Close"] - 1.0
        df = df.dropna()
        return df

    feature_cols = [
        "Close", "Volume", "ret_1",
        "ma_3", "ma_5", "ma_10",
        "std_3", "std_5", "std_10",
        "ret_3", "ret_5", "ret_10",
    ]

    def _generate_realistic_synthetic(n_days: int = 2500):
        """
        Generate realistic synthetic stock data using geometric Brownian motion.
        This produces price/volume series with properties similar to real equities.
        """
        np.random.seed(42)
        # Geometric Brownian Motion for Close price
        S0 = 120.0              # initial price (like AAPL around 2015)
        mu = 0.0003             # daily drift  (~7.5% annualised)
        sigma = 0.015           # daily volatility (~24% annualised)
        dt = 1.0
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
        close = S0 * np.cumprod(1 + returns)
        close = np.insert(close, 0, S0)[:n_days]

        # Synthetic volume correlated with absolute returns
        base_vol = 5e7
        vol_noise = np.random.lognormal(0, 0.3, n_days)
        volume = base_vol * vol_noise * (1 + 5 * np.abs(returns[:n_days]))

        dates = pd.date_range("2015-01-02", periods=n_days, freq="B")
        df = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)
        console.print(
            f"[cyan]Generated realistic synthetic stock data: "
            f"{n_days} trading days, price range "
            f"${df['Close'].min():.2f}–${df['Close'].max():.2f}[/]"
        )
        return df

    def load_stock_data(
        ticker: str = "AAPL",
        start: str = "2015-01-01",
        end: str = "2024-12-31",
    ):
        """
        Load daily OHLCV data for a single ticker.
        Priority:
          1) Local CSV cache  (data/<ticker>.csv)
          2) yfinance download (auto-cached on success)
          3) Realistic synthetic data (geometric Brownian motion)
        """
        cache_dir = Path("data")
        cache_file = cache_dir / f"{ticker}.csv"

        # ---- try local cache first ----
        if cache_file.exists():
            console.print(f"[green]Loading cached data from {cache_file}[/]")
            raw = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(raw) >= 100:
                df = _build_features(raw)
                if len(df) >= 100:
                    console.print(f"  [dim]{len(df)} samples loaded from cache[/]")
                    return df[feature_cols].astype(np.float32).values, df["target"].astype(np.float32).values

        # ---- try yfinance ----
        raw = None
        try:
            console.print(f"[blue]Downloading {ticker} data via yfinance ({start} ~ {end})...[/]")
            raw = yf.download(ticker, start=start, end=end, progress=False)
        except Exception as e:
            console.print(f"[yellow]yfinance download failed: {e}[/]")

        if raw is not None and not raw.empty and len(raw) >= 100:
            # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            # Save to cache for future runs
            cache_dir.mkdir(parents=True, exist_ok=True)
            raw.to_csv(cache_file)
            console.print(f"[green]Downloaded {len(raw)} rows → cached to {cache_file}[/]")
            df = _build_features(raw)
            if len(df) >= 100:
                return df[feature_cols].astype(np.float32).values, df["target"].astype(np.float32).values

        # ---- fallback: realistic synthetic data ----
        console.print("[yellow]Using realistic synthetic stock data (geometric Brownian motion).[/]")
        raw_syn = _generate_realistic_synthetic(2500)
        # Also cache synthetic data so the user can inspect it
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_syn.to_csv(cache_file)
        console.print(f"[dim]Synthetic data cached to {cache_file}[/]")
        df = _build_features(raw_syn)
        X = df[feature_cols].astype(np.float32).values
        y = df["target"].astype(np.float32).values
        return X, y

    X, y = load_stock_data(ticker="AAPL")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # --- 2. Config with LightGBM-oriented search space ---
    config = Config.from_dict(
        {
            "experiment": {
                "name": study_name,
                "task": "regression",
                "objective": "val_rmse",
                "direction": "minimize",
            },
            "tuning": {
                "strategy": "bayesian",
                "n_trials": n_trials,
                "search_space": {
                    "num_leaves": {"type": "int", "low": 16, "high": 256},
                    "learning_rate": {
                        "type": "loguniform",
                        "low": 1e-3,
                        "high": 1e-1,
                    },
                    "max_depth": {"type": "int", "low": 3, "high": 12},
                    "feature_fraction": {
                        "type": "float",
                        "low": 0.6,
                        "high": 1.0,
                    },
                    "bagging_fraction": {
                        "type": "float",
                        "low": 0.6,
                        "high": 1.0,
                    },
                    "bagging_freq": {"type": "int", "low": 1, "high": 10},
                    "min_data_in_leaf": {
                        "type": "int",
                        "low": 10,
                        "high": 200,
                    },
                },
            },
        }
    )

    # --- 3. Objective function using Trial API ---
    def objective(trial: Trial) -> float:
        """
        Objective function for tuning LightGBM on stock data.

        NOTE:
            - We set feature_pre_filter=False to allow changing min_data_in_leaf
              across trials without LightGBM raising errors.
            - RMSE is computed manually to avoid version differences in
              sklearn's mean_squared_error signature.
        """
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            # Allow dynamic min_data_in_leaf changes
            "feature_pre_filter": False,
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 1e-1, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.6, 1.0
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.6, 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 10, 200
            ),
        }

        num_boost_round = 500
        early_stopping_rounds = 50

        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
            ],
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        # Manually compute RMSE to avoid sklearn API differences
        diff = y_pred - y_val
        rmse = float(np.sqrt(np.mean(diff * diff)))

        # Report best metric to trial (optional, useful for pruning / logging)
        trial.report(rmse, step=int(model.best_iteration or 0))

        return rmse

    # --- 4. Run optimization and save Study to studies/ ---
    tuner = Tuner(config, verbose=True)
    study = tuner.optimize(objective, n_trials=n_trials)

    console.print("\n[bold green]Optimization finished[/]")
    console.print(f"  Best RMSE: {study.best_value}")
    console.print(f"  Best params:")
    for k, v in (study.best_params or {}).items():
        console.print(f"    {k}: {v}")

    # --- 5. Retrain with best params and save model ---
    best_params = study.best_params
    if best_params:
        from datetime import datetime as _dt
        console.print("\n[bold blue]Retraining with best parameters...[/]")
        final_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            **best_params,
        }
        final_model = lgb.train(
            final_params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{study_name}_best_{ts}.txt"
        final_model.save_model(str(model_path))
        study.best_model_path = str(model_path)

        # Also save as pickle for scikit-learn compatible loading
        import pickle
        pkl_path = models_dir / f"{study_name}_best_{ts}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(final_model, f)

        console.print(f"  [green]Best model saved to:[/]")
        console.print(f"    LightGBM native: {model_path}")
        console.print(f"    Pickle:          {pkl_path}")
        console.print(f"    Best iteration:  {final_model.best_iteration}")

        # Evaluate the final model
        y_pred_final = final_model.predict(X_val, num_iteration=final_model.best_iteration)
        diff_final = y_pred_final - y_val
        final_rmse = float(np.sqrt(np.mean(diff_final * diff_final)))
        console.print(f"    Final val RMSE:  {final_rmse}")

    studies_dir = Path("studies")
    studies_dir.mkdir(exist_ok=True)
    study_path = studies_dir / f"{study_name}.json"
    study.save(study_path)
    console.print(f"\n[bold]Study saved to:[/] {study_path}")
    console.print(
        "You can open the dashboard and view this study via the Study view."
    )

    # Also record a summary experiment row in SQLite so that the
    # Experiments list in the dashboard has something to display.
    try:
        from mltune.tracker.backend import SQLiteBackend

        backend = SQLiteBackend("mltune.db")
        backend.save_experiment(
            experiment_id=f"exp_{study_name}",
            data={
                "name": study_name,
                "status": "completed",
                "config": config.model_dump(),
                "tags": ["lgbm", "stock"],
                "best_metric": study.best_value,
                "best_step": None,
            },
        )
        console.print(
            f"[green]Experiment exp_{study_name} recorded in SQLite (mltune.db).[/]"
        )
    except Exception as e:
        console.print(
            f"[yellow]Warning: failed to record experiment in SQLite: {e}[/]"
        )

@main.command()
@click.argument("study_path", type=click.Path(exists=True))
def report(study_path: str):
    """Generate a report from a study file."""
    from mltune.optim.study import Study
    from mltune.tracker.visualizer import Visualizer
    
    study = Study.load(study_path)
    
    console.print(f"\n[bold blue]Study Report: {study.study_name}[/]")
    console.print("=" * 50)
    
    summary = study.summary()
    console.print(f"  Direction: {summary.direction}")
    console.print(f"  Total trials: {summary.n_trials}")
    console.print(f"  Completed: {summary.n_completed_trials}")
    console.print(f"  Failed: {summary.n_failed_trials}")
    
    if study.best_value is not None:
        console.print(f"\n[bold green]Best Value: {study.best_value}[/]")
    
    if study.best_params:
        console.print("\n[bold]Best Parameters:[/]")
        for key, value in study.best_params.items():
            console.print(f"  {key}: {value}")
    
    # Parameter importance
    importance = study.param_importance()
    if importance:
        console.print("\n[bold]Parameter Importance:[/]")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for param, score in sorted_importance:
            bar = "█" * int(score * 20)
            console.print(f"  {param}: {score:.3f} {bar}")
    
    # Generate visualizations
    viz = Visualizer()
    
    # Save optimization history plot
    history = study.get_optimization_history()
    if history:
        viz.plot_optimization_history(
            history,
            title=f"Optimization History: {study.study_name}",
            save_path="optimization_history.png",
        )
        console.print("\n[green]Saved: optimization_history.png[/]")


# ---------------------------------------------------------------------------
# Agent-based optimization command (Level 1)
# ---------------------------------------------------------------------------
@main.command(name="agent-tune")
@click.option("--n-trials", "-n", default=20, help="Number of optimization trials")
@click.option("--study-name", default="agent_tune", help="Study name")
@click.option("--model", default=None, help="LLM model name (default: ark-code-latest)")
@click.option("--base-url", default=None, help="Custom API base URL (default: Ark endpoint)")
@click.option("--temperature", default=0.7, help="LLM sampling temperature")
def agent_tune(n_trials: int, study_name: str, model: str, base_url: str, temperature: float):
    """
    Run LLM-agent-driven hyperparameter tuning (Level 1).

    The AI agent reviews past trial history and proposes the next
    set of hyperparameters — replacing TPE with LLM reasoning.

    Default LLM: Volcengine Ark (ark-code-latest).
    Override with --model / --base-url or env vars:
    ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL

    Example:
        mltune agent-tune --n-trials 20
    """
    import numpy as np

    console.print("[bold blue]Running LLM Agent Hyperparameter Tuning[/]")

    from mltune import Config
    from mltune.optim.agent import AgentOptimizer
    from mltune.optim.base import Trial

    # Reuse the LightGBM stock example as a demo objective
    # (user can replace with their own objective)
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
    except ImportError:
        console.print(
            "[red]This demo requires: pip install lightgbm scikit-learn[/]"
        )
        raise click.Abort()

    # Load data (reuse the same logic as lgbm-stock)
    console.print("[dim]Loading data for agent tuning demo...[/]")
    data_file = Path("data/AAPL.csv")
    if data_file.exists():
        import pandas as pd
        raw = pd.read_csv(data_file, index_col=0, parse_dates=True)
        df = raw[["Close", "Volume"]].copy().dropna()
        df["ret_1"] = df["Close"].pct_change()
        for k in (3, 5, 10):
            df[f"ma_{k}"] = df["Close"].rolling(k).mean()
            df[f"std_{k}"] = df["Close"].rolling(k).std()
            df[f"ret_{k}"] = df["Close"].pct_change(k)
        df["target"] = df["Close"].shift(-1) / df["Close"] - 1.0
        df = df.dropna()
        feature_cols = ["Close", "Volume", "ret_1", "ma_3", "ma_5", "ma_10",
                        "std_3", "std_5", "std_10", "ret_3", "ret_5", "ret_10"]
        X = df[feature_cols].astype(np.float32).values
        y = df["target"].astype(np.float32).values
    else:
        console.print("[yellow]No cached data found. Run 'mltune lgbm-stock' first to generate data.[/]")
        raise click.Abort()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    config = Config.from_dict({
        "experiment": {
            "name": study_name,
            "task": "regression",
            "objective": "val_rmse",
            "direction": "minimize",
        },
        "tuning": {
            "strategy": "agent",
            "n_trials": n_trials,
            "search_space": {
                "num_leaves": {"type": "int", "low": 16, "high": 256},
                "learning_rate": {"type": "loguniform", "low": 1e-3, "high": 1e-1},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
                "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
                "bagging_freq": {"type": "int", "low": 1, "high": 10},
                "min_data_in_leaf": {"type": "int", "low": 10, "high": 200},
            },
        },
    })

    # model / base_url = None → AgentOptimizer falls back to Ark defaults
    kwargs: dict = {"temperature": temperature}
    if model:
        kwargs["model"] = model
    if base_url:
        kwargs["base_url"] = base_url

    optimizer = AgentOptimizer(config, **kwargs)

    def objective(trial: Trial) -> float:
        params = trial.params
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            **params,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
        mdl = lgb.train(
            lgb_params, train_set,
            num_boost_round=200,
            valid_sets=[val_set],
            callbacks=callbacks,
        )
        y_pred = mdl.predict(X_val)
        diff = y_pred - y_val
        rmse = float(np.sqrt(np.mean(diff * diff)))
        return rmse

    study = optimizer.optimize(objective, n_trials=n_trials)

    console.print(f"\n[bold green]Agent Tuning Complete[/]")
    console.print(f"  Best RMSE: {study.best_value}")
    if study.best_params:
        console.print(f"  Best params:")
        for k, v in study.best_params.items():
            console.print(f"    {k}: {v}")

    # Retrain with best params and save model
    best_params = study.best_params
    if best_params:
        from datetime import datetime as _dt
        console.print("\n[bold blue]Retraining with best parameters...[/]")
        final_lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            **best_params,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
        final_model = lgb.train(
            final_lgb_params, train_set,
            num_boost_round=200,
            valid_sets=[val_set],
            callbacks=callbacks,
        )
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{study_name}_best_{ts}.txt"
        final_model.save_model(str(model_path))
        study.best_model_path = str(model_path)

        import pickle
        pkl_path = models_dir / f"{study_name}_best_{ts}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(final_model, f)

        y_pred_final = final_model.predict(X_val)
        diff_final = y_pred_final - y_val
        final_rmse = float(np.sqrt(np.mean(diff_final * diff_final)))
        console.print(f"  [green]Best model saved to: {model_path}[/]")
        console.print(f"  [green]Pickle saved to:     {pkl_path}[/]")
        console.print(f"  Final val RMSE:    {final_rmse}")

    studies_dir = Path("studies")
    studies_dir.mkdir(exist_ok=True)
    study.save(studies_dir / f"{study_name}.json")
    console.print(f"\n[bold]Study saved to: studies/{study_name}.json[/]")


# ---------------------------------------------------------------------------
# AutoResearch command (Level 2)
# ---------------------------------------------------------------------------
@main.command(name="autoresearch")
@click.option("--train-script", "-t", default="train.py", help="Training script to modify")
@click.option("--program-md", "-p", default="program.md", help="Research instructions file")
@click.option("--metric", "-m", default="val_loss", help="Metric name to extract from output")
@click.option("--direction", "-d", default="minimize", type=click.Choice(["minimize", "maximize"]))
@click.option("--max-iters", "-n", default=50, help="Maximum experiment iterations")
@click.option("--time-budget", default=300, help="Time budget per training run (seconds)")
@click.option("--model", default=None, help="LLM model name (default: ark-code-latest)")
@click.option("--base-url", default=None, help="Custom API base URL (default: Ark endpoint)")
@click.option("--study-name", default="autoresearch", help="Study name for dashboard")
def autoresearch_cmd(
    train_script: str,
    program_md: str,
    metric: str,
    direction: str,
    max_iters: int,
    time_budget: int,
    model: str,
    base_url: str,
    study_name: str,
):
    """
    Run autonomous AI research (autoresearch style).

    The AI agent reads a program.md with research instructions and the
    training script, proposes code modifications, executes training, and
    keeps or discards changes based on the metric.

    Inspired by karpathy/autoresearch.

    Default LLM: Volcengine Ark (ark-code-latest).
    Override with --model / --base-url or env vars:
    ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL

    Example:
        mltune autoresearch -t train.py -p program.md -m val_bpb -d minimize -n 50
    """
    from mltune.optim.agent import AutoResearchRunner

    train_path = Path(train_script)
    if not train_path.exists():
        console.print(f"[red]Training script not found: {train_script}[/]")
        console.print(
            "Create your training script first, or use the example:\n"
            "  - The script should print the metric in the format: metric_name: value"
        )
        raise click.Abort()

    runner_kwargs: dict = {
        "train_script": train_script,
        "program_md": program_md,
        "eval_metric_name": metric,
        "direction": direction,
        "study_name": study_name,
    }
    if model:
        runner_kwargs["model"] = model
    if base_url:
        runner_kwargs["base_url"] = base_url

    runner = AutoResearchRunner(**runner_kwargs)

    study = runner.run(
        max_iterations=max_iters,
        time_budget_per_run=time_budget,
    )

    # Record in SQLite
    try:
        from mltune.tracker.backend import SQLiteBackend
        backend = SQLiteBackend("mltune.db")
        backend.save_experiment(
            experiment_id=f"exp_{study_name}",
            data={
                "name": study_name,
                "status": "completed",
                "tags": ["autoresearch", "agent"],
                "best_metric": study.best_value,
            },
        )
        console.print(f"[green]Experiment exp_{study_name} recorded in SQLite.[/]")
    except Exception as e:
        console.print(f"[yellow]Warning: SQLite recording failed: {e}[/]")


if __name__ == "__main__":
    main()
