# AutoForge

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/kanlishiyi/autoforge/ci.yml?branch=main&label=CI)](https://github.com/kanlishiyi/autoforge/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/autoforge-mltune.svg)](https://pypi.org/project/autoforge-mltune/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/autoforge-mltune)](https://pypi.org/project/autoforge-mltune/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Intelligent Machine Learning Hyperparameter Tuning Platform**

[English](README.md) | [中文](README_zh.md)

</div>

---

## Screenshots

> Replace these placeholders with real dashboard captures (recommended size: 1600x900).

### Experiments Overview

![AutoForge Dashboard Overview](https://via.placeholder.com/1600x900/10131a/39ffea?text=AutoForge+Dashboard+Overview)

### Study Detail (Live Metrics)

![AutoForge Study Live Metrics](https://via.placeholder.com/1600x900/10131a/ff3ef8?text=AutoForge+Study+Live+Metrics)

---

## Overview

AutoForge is an intelligent hyperparameter tuning platform inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). It translates the idea of autonomous AI-driven research into a practical tuning toolkit:

- 🤖 **AI Agent Optimization** — LLM-driven hyperparameter search + autoresearch-style autonomous code modification
- 🎯 **Multiple Tuning Strategies** — Bayesian (TPE), Random Search, Grid Search, AI Agent
- 📊 **Experiment Tracking** — SQLite backend, full lifecycle management, real-time monitoring
- 🔧 **Flexible Configuration** — YAML / Dict config system with search space definitions
- 🌐 **Web Dashboard** — React + FastAPI dark cyberpunk-themed UI with live polling
- 📈 **Built-in Examples** — LightGBM stock price prediction tuning, ready to use
- 🏆 **Best Model Saving** — Automatically saves the best model with timestamped versioning

## Installation

### From Source

```bash
git clone https://github.com/kanlishiyi/autoforge.git
cd autoforge
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# LightGBM stock example
pip install lightgbm scikit-learn pandas yfinance

# AI Agent optimization (Level 1 & Level 2)
pip install openai

# Dashboard frontend build
cd dashboard && npm install && npm run build && cd ..
```

## Quick Start

### End-to-End in 5 Minutes

```bash
# 1. Install
git clone https://github.com/kanlishiyi/autoforge.git && cd autoforge
pip install -e .
pip install lightgbm scikit-learn pandas yfinance

# 2. Run LightGBM stock tuning (30 trials of Bayesian optimization)
mltune lgbm-stock --n-trials 30 --study-name lgbm_stock_price

# 3. Build and start the Dashboard
cd dashboard && npm install && npm run build && cd ..
mltune dashboard --port 8000

# 4. Open browser
#    Experiments list: http://localhost:8000/dashboard/#/
#    Study details:    http://localhost:8000/dashboard/#/studies/lgbm_stock_price
```

### Python API

```python
from mltune import Tuner, Config

# Define configuration
config = Config.from_dict({
    "experiment": {
        "name": "my_experiment",
        "objective": "val_loss",
        "direction": "minimize",
    },
    "tuning": {
        "strategy": "bayesian",   # bayesian / random / grid / agent
        "n_trials": 50,
        "search_space": {
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "num_layers": {"type": "int", "low": 2, "high": 12},
        },
    },
})

# Define objective function
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    # ... train your model ...
    return val_loss

# Run optimization
tuner = Tuner(config)
study = tuner.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best value:  {study.best_value}")
print(f"Best model:  {study.best_model_path}")
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `mltune lgbm-stock` | LightGBM stock price prediction tuning example |
| `mltune agent-tune` | LLM Agent-driven hyperparameter search |
| `mltune autoresearch` | Autoresearch-style autonomous code modification loop |
| `mltune dashboard` | Launch Web Dashboard |
| `mltune experiments` | List experiments |
| `mltune report <study.json>` | Generate report from a Study file |

### lgbm-stock — LightGBM Tuning Example

```bash
# Bayesian optimization (TPE) for LightGBM hyperparameters
# Data priority: local cache (data/AAPL.csv) → yfinance download → synthetic data (GBM)
mltune lgbm-stock --n-trials 30 --study-name lgbm_stock_price
```

Search space includes 7 hyperparameters: `num_leaves`, `learning_rate`, `max_depth`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `min_data_in_leaf`.

After completion:
- Saves Study to `studies/lgbm_stock_price.json`
- Records experiment to SQLite (`mltune.db`)
- Saves best model to `models/lgbm_stock_price_best_<timestamp>.txt`
- Viewable in Dashboard in real-time

### agent-tune — LLM-Driven Hyperparameter Search (Level 1)

Uses a large language model instead of TPE as the hyperparameter suggestion strategy. The LLM analyzes historical trial results and reasons about the next set of hyperparameters.

**Uses Volcengine Ark (`ark-code-latest`) by default — no extra configuration needed:**

```bash
# Run directly (uses built-in Ark endpoint)
mltune agent-tune --n-trials 20

# Use local Ollama (override default endpoint)
mltune agent-tune --n-trials 20 --model llama3 --base-url http://localhost:11434/v1

# Use other OpenAI-compatible APIs (DeepSeek, vLLM, etc.)
mltune agent-tune --n-trials 20 --model deepseek-chat --base-url https://api.deepseek.com/v1
```

You can also override the default LLM config via environment variables:

```bash
export ANTHROPIC_AUTH_TOKEN="your-api-key"
export ANTHROPIC_BASE_URL="https://your-endpoint/v1"
export ANTHROPIC_MODEL="your-model"
```

### autoresearch — Autonomous Code Modification Loop (Level 2)

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The AI Agent directly modifies the training script:

```bash
# Using default Ark endpoint
mltune autoresearch \
    --train-script train.py \
    --program-md program.md \
    --metric val_loss \
    --direction minimize \
    --max-iters 50 \
    --time-budget 300

# Or specify another model
mltune autoresearch -t train.py -p program.md --model gpt-4o --base-url https://api.openai.com/v1
```

Agent loop:
1. Read `program.md` (research instructions) + experiment history
2. Propose **one** targeted code modification
3. Execute training (fixed time budget)
4. Metric improved → keep changes; otherwise → revert code
5. Record to Study / Dashboard, save best script to `models/<study_name>_best_<timestamp>.py`

## Optimization Strategies

| Strategy | Config Value | Implementation | Use Case |
|----------|-------------|----------------|----------|
| **Bayesian (TPE)** | `bayesian` / `tpe` | Optuna TPESampler | General-purpose, efficient |
| **Random Search** | `random` | Uniform random sampling | Baseline, high-dimensional |
| **Grid Search** | `grid` | Exhaustive enumeration | Few params, discrete space |
| **AI Agent** | `agent` | LLM-inferred suggestions | Leverage LLM reasoning |

Switch via the `tuning.strategy` field in `Config`:

```python
config = Config.from_dict({
    "tuning": {"strategy": "agent", "n_trials": 20, ...}
})
tuner = Tuner(config)
study = tuner.optimize(objective)
```

## Web Dashboard

The Dashboard is built with **React + Vite**, featuring a dark cyberpunk/geek theme, served by a FastAPI backend for data APIs and static files.

### UI Features

- **Dark Geek Theme** — Deep dark background + neon cyan/green accents + grid overlay + glow effects
- **Experiments List** — View all experiments with ID, name, status (glowing capsule badges), best metric, model path
- **Study Detail Panel** — Real-time status indicator (running/completed), progress bar, metric cards
- **Optimization History Chart** — Gradient-filled line chart with neon glow, hover tooltips
- **Trials Detail Table** — All trials with state, value, duration, params; best trial highlighted with ★
- **Best Model Path** — Displays the saved best model file location
- **Parameter Importance** — Cyan→green gradient bar chart showing each hyperparameter's impact
- **Live Polling** — Auto-refreshes every 3 seconds, with manual pause/resume
- **Adaptive Precision** — Y-axis labels and tooltips auto-adjust decimal places based on value range

### Build & Launch

```bash
# Build frontend (once; rebuild after code changes)
cd dashboard
npm install
npm run build
cd ..

# Start Dashboard (must run from project root)
mltune dashboard --port 8000
```

Open your browser at `http://localhost:8000/dashboard/#/`

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/experiments` | List experiments (includes best_model_path) |
| GET | `/experiments/{id}` | Experiment detail |
| GET | `/experiments/{id}/metrics` | Experiment metric data |
| GET | `/studies/{name}` | Study results (includes history, best_model_path) |
| GET | `/studies/{name}/importance` | Parameter importance |
| GET | `/studies/{name}/trials` | All trial details (state, value, duration, params) |

## Real-Time Monitoring

AutoForge supports real-time monitoring during training. The optimizer incrementally saves the Study JSON after each trial, and the Dashboard polls for the latest data automatically:

```
Optimization in progress...
  ┌──────────────────────────────────────────────────┐
  │ ● RUNNING    12 completed / 0 failed / 30 total  │
  │ ████████████░░░░░░░░░░░░  40%                    │
  │                                                  │
  │ Best Value: 1.0001328    Completed: 12  Failed: 0│
  └──────────────────────────────────────────────────┘
```

Incremental save mechanism:
- `Tuner.optimize()` — saves to `studies/<name>.json` after each trial
- `AutoResearchRunner.run()` — saves after each iteration
- Dashboard polls `/studies/{name}` and `/studies/{name}/trials` every 3 seconds

## Model Saving

After tuning, AutoForge automatically saves the best model (timestamped to prevent overwrites):

| Command | Format | Path Example |
|---------|--------|-------------|
| `lgbm-stock` | `.txt` + `.pkl` | `models/lgbm_stock_price_best_20260314_103000.txt` |
| `agent-tune` | `.txt` + `.pkl` | `models/agent_lgbm_best_20260314_103000.txt` |
| `autoresearch` | `.py` (best script) | `models/autoresearch_best_20260314_103000.py` |

Model paths are recorded in the Study and displayed in the Dashboard's experiment list and Study detail page.

## Project Structure

```
autoforge/
├── mltune/                     # Python package
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (all commands)
│   ├── core/                   # Core abstractions
│   │   ├── config.py           # Config management (YAML / Dict)
│   │   ├── experiment.py       # Experiment lifecycle
│   │   └── registry.py         # Component registry
│   ├── optim/                  # Optimization engine
│   │   ├── base.py             # BaseOptimizer + Trial interface
│   │   ├── bayesian.py         # Bayesian optimization (Optuna TPE)
│   │   ├── grid.py             # Grid search + Random search
│   │   ├── agent.py            # AgentOptimizer + AutoResearchRunner
│   │   ├── study.py            # Study (trial collection + incremental save)
│   │   └── tuner.py            # Tuner high-level interface
│   ├── tracker/                # Experiment tracking
│   │   ├── backend.py          # SQLite storage backend
│   │   ├── metrics.py          # Metric collection
│   │   └── visualizer.py       # Visualization tools
│   ├── api/                    # Web API
│   │   └── routes.py           # FastAPI routes + static file mount
│   └── utils/                  # Utilities
│       ├── common.py
│       ├── device.py
│       └── seed.py
├── dashboard/                  # Web Dashboard (React + Vite)
│   ├── src/
│   │   ├── App.tsx             # Routing & layout
│   │   ├── api.ts              # Backend API calls
│   │   ├── styles.css          # Dark cyberpunk theme
│   │   └── pages/
│   │       ├── ExperimentsList.tsx  # Experiments list (glowing status badges)
│   │       ├── ExperimentDetail.tsx # Experiment detail + metric charts
│   │       └── StudyView.tsx       # Study detail (live monitoring + charts)
│   ├── vite.config.ts
│   └── package.json
├── examples/                   # Example scripts
│   ├── simple_optimization.py  # Basic optimization example
│   └── agent_optimization.py   # AI Agent optimization example
├── train.py                    # AutoResearch example training script
├── program.md                  # AutoResearch research instructions
├── configs/
│   └── example_config.yaml     # Config template
├── data/                       # Cached data (e.g. AAPL.csv)
├── models/                     # Best model save directory (timestamped)
├── studies/                    # Study JSON output (incremental save)
├── logs/                       # Logs
├── tests/                      # Tests
├── pyproject.toml              # Project dependencies & build config
├── README.md                   # English documentation
└── README_zh.md                # 中文文档
```

## Data Flow

```
                    ┌──────────────┐
                    │  Config      │  YAML / Dict configuration
                    │  (strategy,  │
                    │  search_space)│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Tuner      │  High-level interface, selects optimizer
                    └──────┬───────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
     ┌────────▼──┐  ┌──────▼────┐ ┌──────▼──────┐
     │ Bayesian  │  │  Random/  │ │   Agent     │
     │ (TPE)     │  │  Grid     │ │ (LLM)      │
     └────────┬──┘  └──────┬────┘ └──────┬──────┘
              │            │             │
              └────────────┼─────────────┘
                           │
                    ┌──────▼───────┐
                    │  objective() │  User-defined training function
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Study      │  Collects all Trial results
                    │  (incremental│  Writes JSON after each trial
                    │   save)      │
                    └──────┬───────┘
                           │
         ┌─────────┬───────┼───────┬─────────┐
         │         │       │       │         │
    ┌────▼──┐ ┌────▼──┐ ┌──▼───┐ ┌▼──────┐ ┌▼─────────┐
    │ JSON  │ │SQLite │ │Models│ │ API   │ │Dashboard │
    │(study)│ │(.db)  │ │(.txt │ │(Fast  │ │(React)   │
    │       │ │       │ │.pkl  │ │ API)  │ │Dark Geek │
    │       │ │       │ │.py)  │ │       │ │Live Poll │
    └───────┘ └───────┘ └──────┘ └───────┘ └──────────┘
```

## Advanced Usage

### Custom Optimizer

```python
from mltune.optim import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def suggest(self, trial):
        """Suggest parameters for the current trial."""
        return {"lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True)}

    def tell(self, trial, value):
        """Receive trial result."""
        pass

optimizer = MyOptimizer(config)
study = optimizer.optimize(objective, n_trials=100)
```

### AgentOptimizer Python API

```python
from mltune.optim.agent import AgentOptimizer

# Uses Ark (ark-code-latest) by default — no LLM params needed
optimizer = AgentOptimizer(config, temperature=0.7, fallback_to_random=True)
study = optimizer.optimize(objective, n_trials=20)

# Override with another endpoint
optimizer = AgentOptimizer(
    config,
    model="gpt-4o-mini",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)
```

### AutoResearchRunner Python API

```python
from mltune.optim.agent import AutoResearchRunner

# Uses Ark (ark-code-latest) by default
runner = AutoResearchRunner(
    train_script="train.py",
    program_md="program.md",
    eval_metric_name="val_loss",
    direction="minimize",
)
study = runner.run(max_iterations=50, time_budget_per_run=300)

# Best script saved to models/autoresearch_best_<timestamp>.py
print(f"Best model: {study.best_model_path}")
```

## Development

```bash
# Clone and install
git clone https://github.com/kanlishiyi/autoforge.git
cd autoforge
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black mltune tests
isort mltune tests

# Type checking
mypy mltune
```

### CI and Release (GitHub Actions)

- CI workflow: `.github/workflows/ci.yml`
- PyPI publish workflow: `.github/workflows/publish-pypi.yml`

Release setup checklist:

1. On GitHub, configure **PyPI Trusted Publisher** for this repository.
2. In GitHub repository settings, create environment `pypi` (optional approval rules).
3. Create a GitHub Release (or run workflow manually) to publish package artifacts to PyPI.

If you prefer token-based publishing, set repository secret `PYPI_API_TOKEN` and update the publish step accordingly.

#### Recommended Guardrails (CI must pass before release)

1. Go to `Settings -> Branches -> Add branch protection rule` for `main`.
2. Enable `Require a pull request before merging`.
3. Enable `Require status checks to pass before merging`.
4. In required checks, select all CI checks from `.github/workflows/ci.yml` (all Python matrix jobs).
5. Enable `Require branches to be up to date before merging`.
6. Optional but recommended: enable `Require approval` and `Dismiss stale pull request approvals`.

Release checklist (recommended):

- [ ] PR merged into `main` with all CI checks green
- [ ] Version updated in `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] GitHub Release created (tag + release notes)
- [ ] Verify package on [PyPI](https://pypi.org/project/autoforge-mltune/)

### Dashboard Dev Mode

```bash
cd dashboard
npm install
npm run dev    # Dev server at http://localhost:5173
npm run build  # Production build to dist/
```

Dashboard tech stack:
- **React 18** + **TypeScript** — Frontend framework
- **Vite** — Build tool (`base: "/dashboard/"` for FastAPI static serving)
- **HashRouter** — Client-side routing, no server config needed
- **SVG** — Chart rendering (optimization history line chart, parameter importance bar chart)
- **Polling** — Polls backend API every 3 seconds for live updates

## Acknowledgements

- Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- Bayesian optimization powered by [Optuna](https://optuna.org/)
- Dashboard UI inspired by [TensorBoard](https://www.tensorflow.org/tensorboard) and [W&B](https://wandb.ai/)

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Examples](examples/)** • 
**[Changelog](CHANGELOG.md)**

</div>
