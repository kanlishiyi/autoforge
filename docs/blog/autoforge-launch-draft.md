# AutoForge: From Hyperparameter Tuning to LLM-Driven Autonomous Research

> Draft post for product launch and technical sharing.

## TL;DR

AutoForge is an open-source platform for machine learning optimization that combines:

- classic HPO methods (Bayesian TPE / random / grid),
- LLM agent-based parameter exploration,
- and autoresearch-style autonomous code iteration.

It also ships with an end-to-end dashboard for real-time study monitoring and best-model tracking.

## Why AutoForge

Most ML teams still treat tuning and experimentation as disconnected workflows:

1. scripts are edited manually,
2. experiments are tracked in ad hoc files,
3. and optimization logic is hard-coded in each project.

AutoForge unifies these into one system with a single CLI and Python API, so teams can move from "run trials" to "continuous research loops."

## Core Architecture

### 1) Study-Centric Optimization Layer

- Unified `Study` abstraction for trials, metrics, and metadata.
- Multiple optimizers share the same data model.
- Incremental persistence after each trial enables live dashboard updates.

### 2) Agent Optimization (Level 1)

- LLM proposes candidate hyperparameters based on trial history.
- Existing objective function remains unchanged.
- Great for narrowing search under constrained budgets.

### 3) AutoResearch (Level 2)

- LLM can revise training script variants directly.
- Each candidate script is executed, measured, and accepted/rejected by objective.
- Best code is saved with timestamps for reproducibility.

## Real-Time Experiment Visibility

The Web Dashboard provides:

- optimization history,
- best-parameter and importance views,
- trial-level details (state / duration / value / params),
- and best model path for quick artifact retrieval.

This shortens iteration loops for both engineers and researchers.

## Practical Workflow

```bash
# 1) install
pip install -e ".[dev]"

# 2) run example tuning
mltune lgbm-stock --n-trials 30 --study-name lgbm_stock_price

# 3) launch dashboard
mltune dashboard --port 8000
```

For autonomous research:

```bash
mltune autoresearch -t train.py -p program.md -m val_loss -d minimize -n 20
```

## Differentiators

- **Pragmatic first**: real code execution + measurable objective improvements.
- **Compatible by design**: keep package/CLI stability (`mltune`) while evolving branding (`AutoForge`).
- **Ops-friendly**: CI and PyPI release workflows included.

## Roadmap

- Multi-agent collaboration (planner / coder / critic).
- Better artifact lineage (dataset + model + prompt version links).
- Optional cloud execution backends for parallel research loops.

## Closing

AutoForge is built for teams that want more than static HPO scripts. If you are exploring LLM-native ML tooling, this is a practical starting point to move from experiments to autonomous research cycles.
