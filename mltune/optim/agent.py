"""
Agent-based optimizer: use LLM to suggest hyperparameters.

This module implements two levels of AI-agent-driven optimization:

Level 1 - AgentOptimizer:
    LLM replaces TPE/Random as the "suggest" strategy. The LLM reviews
    past trial history and proposes the next set of hyperparameters.
    Works within the existing search_space definition.

Level 2 - AutoResearchRunner:
    Inspired by karpathy/autoresearch. The LLM directly modifies a
    training script, runs it, evaluates the result, and keeps or
    discards the change. No predefined search space needed.

Both levels integrate with AutoForge's Study / Dashboard ecosystem.

Default LLM endpoint: Volcengine Ark (ark-code-latest).
Override via constructor args or environment variables:
    ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL / ANTHROPIC_MODEL
"""

from __future__ import annotations

import json
import re
import time
import copy
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mltune.core.config import Config, SearchSpaceParam
from mltune.optim.base import BaseOptimizer, Trial, TrialResult, TrialState
from mltune.optim.study import Study

# ---------------------------------------------------------------------------
# Default LLM endpoint — Volcengine Ark
# Can be overridden via env vars or constructor arguments.
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "ark-code-latest"
DEFAULT_LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v1"
DEFAULT_LLM_API_KEY = "a36c3dfc-dfab-4d14-bcae-b3cab07f5cf8"

# Environment variable names (checked in order of priority)
_ENV_API_KEY = "ANTHROPIC_AUTH_TOKEN"
_ENV_BASE_URL = "ANTHROPIC_BASE_URL"
_ENV_MODEL = "ANTHROPIC_MODEL"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_trial_history_text(
    trials: List[TrialResult],
    max_recent: int = 20,
) -> str:
    """Format recent trial results as human-readable text for the LLM."""
    recent = trials[-max_recent:]
    if not recent:
        return "(no previous trials)"
    lines: list[str] = []
    for t in recent:
        status = t.state.value if hasattr(t.state, "value") else str(t.state)
        val_str = f"{t.value:.6f}" if t.value is not None else "FAILED"
        params_str = json.dumps(t.params, ensure_ascii=False)
        lines.append(f"  Trial #{t.trial_id}: value={val_str} status={status} params={params_str}")
    return "\n".join(lines)


def _build_search_space_text(search_space: Dict[str, SearchSpaceParam]) -> str:
    """Format the search space definition for the LLM."""
    lines: list[str] = []
    for name, p in search_space.items():
        if p.choices:
            lines.append(f"  {name}: categorical, choices={p.choices}")
        elif p.type == "int":
            lines.append(f"  {name}: int, range=[{int(p.low)}, {int(p.high)}], log={p.log}")
        elif p.type in ("float", "loguniform"):
            log = p.log or p.type == "loguniform"
            lines.append(f"  {name}: float, range=[{p.low}, {p.high}], log={log}")
        else:
            lines.append(f"  {name}: {p.type}, low={p.low}, high={p.high}")
    return "\n".join(lines) if lines else "(no search space defined)"


def _extract_json_from_llm(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from LLM response text.
    Handles markdown code fences, trailing commas, etc.
    """
    # Try to find JSON in code fences first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Try to find { ... } block
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        raw = m.group(0)
        # Remove trailing commas before } (common LLM mistake)
        raw = re.sub(r",\s*}", "}", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Level 1: AgentOptimizer — LLM-driven hyperparameter search
# ---------------------------------------------------------------------------

DEFAULT_SUGGEST_PROMPT = """\
You are an expert machine learning hyperparameter tuner.

## Task
Suggest the next set of hyperparameters that will **{direction}** the objective.

## Search Space
{search_space}

## Previous Trials (most recent {max_recent})
{trial_history}

## Current Best
Value: {best_value}
Params: {best_params}

## Instructions
1. Analyze the patterns in previous trials — which parameters correlate with better results?
2. Balance exploration (trying new regions) with exploitation (refining promising areas).
3. Return ONLY a single JSON object with the parameter names as keys and suggested values.
4. All values must be within the defined search space ranges.
5. Do NOT include any explanation, just the JSON.
"""


class AgentOptimizer(BaseOptimizer):
    """
    LLM-driven hyperparameter optimizer.

    Uses a large language model to analyze past trial history and suggest
    the next set of hyperparameters, combining the reasoning ability of
    LLMs with traditional HPO evaluation loops.

    Supports any OpenAI-compatible API (OpenAI, Azure, local vLLM, Ollama, etc.)

    Example:
        ```python
        # Uses Ark endpoint by default; no extra config needed
        optimizer = AgentOptimizer(config)
        study = optimizer.optimize(objective, n_trials=30)

        # Or override with any OpenAI-compatible endpoint
        optimizer = AgentOptimizer(
            config,
            model="gpt-4o-mini",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        )
        ```
    """

    def __init__(
        self,
        config: Config,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: float = 0.7,
        max_recent_trials: int = 20,
        fallback_to_random: bool = True,
    ):
        """
        Args:
            config: AutoForge Config object (must have search_space defined).
            model: LLM model name. Defaults to ANTHROPIC_MODEL env var
                or ``ark-code-latest``.
            api_key: API key. Defaults to ANTHROPIC_AUTH_TOKEN env var
                or built-in Ark key.
            base_url: Custom API base URL. Defaults to ANTHROPIC_BASE_URL
                env var or Ark endpoint.
            prompt_template: Custom prompt template with {direction},
                {search_space}, {trial_history}, {best_value}, {best_params} placeholders.
            temperature: LLM sampling temperature.
            max_recent_trials: How many recent trials to include in the prompt.
            fallback_to_random: If LLM call fails, fall back to random sampling.
        """
        super().__init__(config)
        import os
        self.model = model or os.environ.get(_ENV_MODEL, DEFAULT_LLM_MODEL)
        self.api_key = api_key or os.environ.get(_ENV_API_KEY, DEFAULT_LLM_API_KEY)
        self.base_url = base_url or os.environ.get(_ENV_BASE_URL, DEFAULT_LLM_BASE_URL)
        self.prompt_template = prompt_template or DEFAULT_SUGGEST_PROMPT
        self.temperature = temperature
        self.max_recent_trials = max_recent_trials
        self.fallback_to_random = fallback_to_random

        self._completed_trials: List[TrialResult] = []
        self._client: Any = None  # lazy init

    # ---- LLM client ----

    def _get_client(self):
        """Lazily create the OpenAI-compatible client."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for AgentOptimizer. "
                "Install with: pip install openai"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return self._client

    # ---- Core interface ----

    def suggest(self, trial: Trial) -> Dict[str, Any]:
        """Use LLM to suggest hyperparameters for the next trial."""
        try:
            return self._llm_suggest()
        except Exception as e:
            if self.fallback_to_random:
                import random as _rng
                import numpy as _np
                params: Dict[str, Any] = {}
                for name, p in self.search_space.items():
                    if p.choices:
                        params[name] = _rng.choice(p.choices)
                    elif p.type == "int":
                        params[name] = _rng.randint(int(p.low), int(p.high))
                    elif p.type in ("float", "loguniform"):
                        if p.log or p.type == "loguniform":
                            params[name] = float(_np.exp(_rng.uniform(_np.log(p.low), _np.log(p.high))))
                        else:
                            params[name] = _rng.uniform(p.low, p.high)
                return params
            raise

    def tell(self, trial: Trial, value: float) -> None:
        """Record trial result for future LLM context."""
        if trial.result:
            self._completed_trials.append(trial.result)

    def _llm_suggest(self) -> Dict[str, Any]:
        """Call the LLM and parse the suggested parameters."""
        # Build the prompt
        best_value = None
        best_params = None
        direction = self.config.experiment.direction

        if self._completed_trials:
            completed = [t for t in self._completed_trials if t.value is not None]
            if completed:
                if direction == "minimize":
                    best_trial = min(completed, key=lambda t: t.value)
                else:
                    best_trial = max(completed, key=lambda t: t.value)
                best_value = best_trial.value
                best_params = best_trial.params

        prompt = self.prompt_template.format(
            direction=direction,
            search_space=_build_search_space_text(self.search_space),
            trial_history=_build_trial_history_text(self._completed_trials, self.max_recent_trials),
            best_value=f"{best_value:.6f}" if best_value is not None else "N/A (first trial)",
            best_params=json.dumps(best_params, ensure_ascii=False) if best_params else "N/A",
            max_recent=self.max_recent_trials,
        )

        # Call LLM
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        text = response.choices[0].message.content or ""

        # Parse JSON
        params = _extract_json_from_llm(text)

        # Clamp values to search space bounds
        params = self._clamp_params(params)
        return params

    def _clamp_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure suggested params are within search space bounds."""
        clamped: Dict[str, Any] = {}
        for name, p in self.search_space.items():
            val = params.get(name)
            if val is None:
                # LLM forgot this param — use random
                import random as _rng
                import numpy as _np
                if p.choices:
                    val = _rng.choice(p.choices)
                elif p.type == "int":
                    val = _rng.randint(int(p.low), int(p.high))
                else:
                    val = _rng.uniform(p.low, p.high)
            else:
                if p.choices:
                    if val not in p.choices:
                        import random as _rng
                        val = _rng.choice(p.choices)
                elif p.type == "int":
                    val = int(max(p.low, min(p.high, val)))
                else:
                    val = float(max(p.low, min(p.high, val)))
            clamped[name] = val
        return clamped


# ---------------------------------------------------------------------------
# Level 2: AutoResearchRunner — autoresearch-style code modification loop
# ---------------------------------------------------------------------------

@dataclass
class ExperimentLog:
    """Record of a single autoresearch experiment iteration."""
    iteration: int
    description: str          # Agent's natural language description of the change
    diff: str                 # Code diff (unified)
    metric_value: Optional[float] = None
    accepted: bool = False
    duration_seconds: float = 0.0
    error: Optional[str] = None


class AutoResearchRunner:
    """
    Autonomous research runner inspired by karpathy/autoresearch.

    The LLM agent reads a program.md (research instructions) and the
    current training script, proposes modifications, executes the script,
    and keeps or discards changes based on the evaluation metric.

    This integrates with AutoForge's Study / Dashboard ecosystem so you
    can visualize the progress on the web dashboard.

    Example:
        ```python
        # Uses Ark (ark-code-latest) by default
        runner = AutoResearchRunner(
            train_script="train.py",
            program_md="program.md",
            eval_metric_name="val_bpb",
            direction="minimize",
        )
        runner.run(max_iterations=50, time_budget_per_run=300)
        ```
    """

    def __init__(
        self,
        train_script: str = "train.py",
        program_md: str = "program.md",
        eval_metric_name: str = "val_loss",
        direction: str = "minimize",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        study_name: str = "autoresearch",
        work_dir: str = ".",
    ):
        import os
        self.train_script = Path(work_dir) / train_script
        self.program_md = Path(work_dir) / program_md
        self.eval_metric_name = eval_metric_name
        self.direction = direction
        self.model = model or os.environ.get(_ENV_MODEL, DEFAULT_LLM_MODEL)
        self.api_key = api_key or os.environ.get(_ENV_API_KEY, DEFAULT_LLM_API_KEY)
        self.base_url = base_url or os.environ.get(_ENV_BASE_URL, DEFAULT_LLM_BASE_URL)
        self.temperature = temperature
        self.study_name = study_name
        self.work_dir = Path(work_dir)

        self.experiments: List[ExperimentLog] = []
        self.best_value: Optional[float] = None
        self.best_code: Optional[str] = None
        self._client: Any = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return self._client

    # ---- Main loop ----

    def run(
        self,
        max_iterations: int = 50,
        time_budget_per_run: int = 300,
        python_cmd: Optional[str] = None,
    ) -> Study:
        """
        Run the autonomous research loop.

        Args:
            max_iterations: Maximum number of experiment iterations.
            time_budget_per_run: Wall-clock seconds allowed per training run.
            python_cmd: Python executable to invoke for training.

        Returns:
            AutoForge Study recording all iterations.
        """
        import sys
        if python_cmd is None:
            python_cmd = sys.executable  # use the same Python that runs mltune

        from rich.console import Console
        console = Console()

        # Build AutoForge Study for tracking
        study = Study(
            study_name=self.study_name,
            direction=self.direction,
        )

        console.print(f"\n[bold cyan]{'='*60}[/]")
        console.print(f"[bold cyan]AutoResearch — AI Agent Autonomous Optimization[/]")
        console.print(f"[bold cyan]{'='*60}[/]")
        console.print(f"  Script:     {self.train_script}")
        console.print(f"  Program:    {self.program_md}")
        console.print(f"  Metric:     {self.eval_metric_name} ({self.direction})")
        console.print(f"  Model:      {self.model}")
        console.print(f"  Iterations: {max_iterations}")
        console.print(f"  Time/run:   {time_budget_per_run}s")
        console.print(f"[bold cyan]{'='*60}[/]\n")

        for i in range(max_iterations):
            console.print(f"[bold blue]--- Iteration {i+1}/{max_iterations} ---[/]")

            # Read current state
            current_code = self.train_script.read_text(encoding="utf-8")
            program = self.program_md.read_text(encoding="utf-8") if self.program_md.exists() else ""

            # Backup
            backup_code = current_code

            # Ask agent to propose modification
            try:
                new_code, description = self._agent_propose(
                    current_code, program, i
                )
            except Exception as e:
                console.print(f"  [red]Agent proposal failed: {e}[/]")
                self.experiments.append(ExperimentLog(
                    iteration=i, description=f"PROPOSAL FAILED: {e}", diff="", error=str(e),
                ))
                continue

            console.print(f"  [dim]Change: {description}[/]")

            # Write modified code
            self.train_script.write_text(new_code, encoding="utf-8")

            # Execute training
            t0 = time.time()
            try:
                metric_val = self._run_training(python_cmd, time_budget_per_run)
                duration = time.time() - t0
            except Exception as e:
                duration = time.time() - t0
                console.print(f"  [red]Training failed ({duration:.0f}s): {e}[/]")
                # Revert
                self.train_script.write_text(backup_code, encoding="utf-8")
                self.experiments.append(ExperimentLog(
                    iteration=i, description=description, diff=self._make_diff(backup_code, new_code),
                    error=str(e), duration_seconds=duration,
                ))
                trial_result = TrialResult(
                    trial_id=i, params={"description": description},
                    value=None, state=TrialState.FAILED, error=str(e),
                )
                study.add_trial(trial_result)
                continue

            # Decide keep or discard
            accepted = self._is_improvement(metric_val)
            if accepted:
                self.best_value = metric_val
                self.best_code = new_code
                console.print(f"  [green]✓ ACCEPTED  metric={metric_val:.6f} (new best!) [{duration:.0f}s][/]")
            else:
                # Revert
                self.train_script.write_text(backup_code, encoding="utf-8")
                console.print(
                    f"  [yellow]✗ REJECTED  metric={metric_val:.6f} "
                    f"(best={self.best_value:.6f}) [{duration:.0f}s][/]"
                )

            self.experiments.append(ExperimentLog(
                iteration=i, description=description,
                diff=self._make_diff(backup_code, new_code),
                metric_value=metric_val, accepted=accepted,
                duration_seconds=duration,
            ))

            trial_result = TrialResult(
                trial_id=i,
                params={"description": description, "accepted": accepted},
                value=metric_val,
                state=TrialState.COMPLETED,
                duration=duration,
            )
            study.add_trial(trial_result)

            # Incremental save — so Dashboard can poll live progress
            try:
                _live_dir = self.work_dir / "studies"
                _live_dir.mkdir(exist_ok=True)
                study.save(_live_dir / f"{self.study_name}.json")
            except Exception:
                pass

        # Save best model artifact (the optimized training script)
        if self.best_code is not None:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            models_dir = self.work_dir / "models"
            models_dir.mkdir(exist_ok=True)
            best_script_path = models_dir / f"{self.study_name}_best_{ts}.py"
            best_script_path.write_text(self.best_code, encoding="utf-8")
            study.best_model_path = str(best_script_path)
            console.print(f"  [green]Best script saved to: {best_script_path}[/]")

        # Save study
        studies_dir = self.work_dir / "studies"
        studies_dir.mkdir(exist_ok=True)
        study.save(studies_dir / f"{self.study_name}.json")

        # Print summary
        console.print(f"\n[bold green]{'='*60}[/]")
        console.print(f"[bold green]AutoResearch Complete[/]")
        console.print(f"[bold green]{'='*60}[/]")
        n_accepted = sum(1 for e in self.experiments if e.accepted)
        console.print(f"  Total iterations: {len(self.experiments)}")
        console.print(f"  Accepted changes: {n_accepted}")
        console.print(f"  Best {self.eval_metric_name}: {self.best_value}")
        if study.best_model_path:
            console.print(f"  Best script: {study.best_model_path}")
        console.print(f"[bold green]{'='*60}[/]\n")

        return study

    # ---- Agent proposal ----

    def _agent_propose(
        self, current_code: str, program: str, iteration: int,
    ) -> Tuple[str, str]:
        """Ask the LLM to propose a code modification."""
        history_text = self._format_experiment_history()

        prompt = f"""\
You are an autonomous AI research agent. Your goal is to improve a training script
by making targeted modifications, one change at a time.

## Research Instructions
{program if program else "(none provided)"}

## Experiment History
{history_text}

## Current Best
Metric ({self.eval_metric_name}): {self.best_value if self.best_value is not None else "N/A (first run)"}
Direction: {self.direction} (lower is better) if minimize, (higher is better) if maximize

## Current Code ({self.train_script.name})
```python
{current_code}
```

## Your Task (Iteration {iteration + 1})
1. Analyze the experiment history and current code.
2. Propose ONE targeted modification that you believe will improve the metric.
3. Return your response in this exact format:

DESCRIPTION: <one-line description of what you changed and why>
```python
<the complete modified file content>
```

IMPORTANT:
- Return the COMPLETE file, not just the changed parts.
- Make only ONE logical change per iteration.
- Be creative but grounded — consider architecture, hyperparameters, regularization, etc.
- If a previous change was rejected, try a different approach.
"""

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=8192,
        )
        text = response.choices[0].message.content or ""

        # Parse description
        desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", text)
        description = desc_match.group(1).strip() if desc_match else "Agent modification"

        # Parse code
        code_match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not code_match:
            raise ValueError("Agent did not return a valid code block")
        new_code = code_match.group(1)

        return new_code, description

    # ---- Training execution ----

    def _run_training(self, python_cmd: str, timeout: int) -> float:
        """Execute the training script and extract the metric."""
        result = subprocess.run(
            [python_cmd, str(self.train_script)],
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # grace period
            cwd=str(self.work_dir),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Training script exited with code {result.returncode}.\n"
                f"stderr: {result.stderr[-1000:]}"
            )

        # Extract metric from stdout — look for "metric_name: <value>" pattern
        output = result.stdout + "\n" + result.stderr
        pattern = rf"{re.escape(self.eval_metric_name)}\s*[:=]\s*([\d.eE+-]+)"
        matches = re.findall(pattern, output)
        if not matches:
            raise ValueError(
                f"Could not find '{self.eval_metric_name}' in training output.\n"
                f"Last 500 chars: {output[-500:]}"
            )

        # Take the last reported value
        return float(matches[-1])

    # ---- Helpers ----

    def _is_improvement(self, value: float) -> bool:
        """Check if value is better than current best."""
        if self.best_value is None:
            return True
        if self.direction == "minimize":
            return value < self.best_value
        else:
            return value > self.best_value

    def _format_experiment_history(self) -> str:
        if not self.experiments:
            return "(no previous experiments)"
        lines: list[str] = []
        for e in self.experiments[-15:]:
            status = "✓ ACCEPTED" if e.accepted else "✗ REJECTED"
            val = f"{e.metric_value:.6f}" if e.metric_value is not None else "FAILED"
            lines.append(f"  #{e.iteration}: {status} metric={val} — {e.description}")
        return "\n".join(lines)

    @staticmethod
    def _make_diff(old_code: str, new_code: str) -> str:
        """Generate a unified diff between old and new code."""
        import difflib
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, fromfile="before", tofile="after")
        return "".join(diff)
