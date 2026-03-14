"""
Advanced example: Using AI Agent for optimization.

This example shows two levels of AI-agent-driven optimization:

Level 1 - AgentOptimizer:
    LLM replaces TPE as the hyperparameter suggestion strategy.
    The LLM reviews past trial history and proposes the next
    set of hyperparameters using reasoning.

Level 2 - AutoResearchRunner:
    Inspired by karpathy/autoresearch. The LLM directly modifies
    a training script, runs it, evaluates the result, and keeps
    or discards the change.

Requirements:
    pip install openai lightgbm scikit-learn

Default LLM: Volcengine Ark (ark-code-latest) — no extra config needed.

To use a different endpoint, override via env vars:
    export ANTHROPIC_AUTH_TOKEN=your-key
    export ANTHROPIC_BASE_URL=https://your-endpoint/v1
    export ANTHROPIC_MODEL=your-model

Or pass directly to the constructor:
    AgentOptimizer(config, model="gpt-4o-mini", api_key="sk-...", base_url="...")
"""

from mltune.optim.agent import AgentOptimizer, AutoResearchRunner
from mltune import Config


# ============================================================
# Level 1: LLM-driven hyperparameter search
# ============================================================

def demo_agent_optimizer():
    """
    The LLM acts as a "smart sampler" — it analyzes past results
    and proposes the next set of hyperparameters.
    """
    import math
    import random

    print("=" * 60)
    print("Level 1: LLM-driven Hyperparameter Search")
    print("=" * 60)

    # A simple objective function (simulated)
    def train_model(lr, batch_size, hidden_dim):
        lr_score = math.exp(-10 * (math.log10(lr) - math.log10(0.01)) ** 2)
        bs_score = math.exp(-0.001 * (batch_size - 64) ** 2)
        hd_score = math.exp(-0.0001 * (hidden_dim - 128) ** 2)
        noise = random.gauss(0, 0.02)
        return max(0, min(1, lr_score * bs_score * hd_score + noise))

    config = Config.from_dict({
        "experiment": {
            "name": "agent_optimization",
            "objective": "accuracy",
            "direction": "maximize",
        },
        "tuning": {
            "strategy": "agent",    # <-- Use LLM agent!
            "n_trials": 20,
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "low": 1e-4,
                    "high": 1e-1,
                },
                "batch_size": {
                    "type": "categorical",
                    "choices": [16, 32, 64, 128],
                },
                "hidden_dim": {
                    "type": "int",
                    "low": 32,
                    "high": 256,
                },
            },
        },
    })

    # Option A: Use via Tuner (automatic — just set strategy="agent")
    # tuner = Tuner(config)
    # study = tuner.optimize(objective, n_trials=20)

    # Option B: Use AgentOptimizer directly (more control)
    optimizer = AgentOptimizer(
        config,
        # Uses Ark (ark-code-latest) by default — no extra config needed!
        # model="gpt-4o-mini",           # override for OpenAI
        # api_key="sk-...",              # override for OpenAI
        # base_url="http://localhost:11434/v1",  # for Ollama
        temperature=0.7,
        fallback_to_random=True,        # if LLM fails, fall back to random
    )

    def objective(trial):
        """Objective function — params are pre-filled by the LLM agent."""
        lr = trial.params.get("learning_rate", 0.01)
        batch_size = trial.params.get("batch_size", 32)
        hidden_dim = trial.params.get("hidden_dim", 64)
        return train_model(lr, batch_size, hidden_dim)

    study = optimizer.optimize(objective, n_trials=20)

    print(f"\nBest accuracy: {study.best_value:.4f}")
    print("Best parameters:")
    for param, value in (study.best_params or {}).items():
        print(f"  {param}: {value}")


# ============================================================
# Level 2: AutoResearch — code modification loop
# ============================================================

def demo_autoresearch():
    """
    Inspired by karpathy/autoresearch.

    The LLM agent reads a program.md and the training script, proposes
    code modifications, executes training, and keeps/discards changes.

    Before running, create:
      - train.py  (your training script, should print "val_loss: <value>")
      - program.md (research instructions for the agent)
    """
    print("=" * 60)
    print("Level 2: AutoResearch — Autonomous Code Modification")
    print("=" * 60)

    runner = AutoResearchRunner(
        train_script="train.py",        # The script the agent modifies
        program_md="program.md",        # Research instructions
        eval_metric_name="val_loss",    # Metric to extract from stdout
        direction="minimize",
        # Uses Ark (ark-code-latest) by default
        study_name="autoresearch_demo",
    )

    study = runner.run(
        max_iterations=50,              # Max experiments
        time_budget_per_run=300,        # 5 min per training run
    )

    print(f"\nBest {runner.eval_metric_name}: {study.best_value}")
    print(f"Accepted changes: {sum(1 for e in runner.experiments if e.accepted)}")


# ============================================================
# CLI entry
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--autoresearch":
        demo_autoresearch()
    else:
        demo_agent_optimizer()
