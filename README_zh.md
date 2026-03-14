# AutoForge

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**智能机器学习超参数调优平台**

[English](README.md) | [中文](README_zh.md)

</div>

---

## 概述

AutoForge 是一个智能超参数调优平台，灵感来自 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch)。它将自主 AI 驱动研究的理念转化为实用的调优工具包：

- 🤖 **AI Agent 优化** — LLM 驱动的超参搜索 + autoresearch 风格的自主代码修改
- 🎯 **多种调优策略** — 贝叶斯优化 (TPE)、随机搜索、网格搜索、AI Agent
- 📊 **实验追踪** — SQLite 后端存储，完整生命周期管理，实时监控
- 🔧 **灵活配置** — YAML / Dict 配置系统，支持搜索空间定义
- 🌐 **Web Dashboard** — React + FastAPI 暗色赛博朋克极客风 UI，实时轮询更新
- 📈 **内置示例** — LightGBM 股票价格预测调优，开箱即用
- 🏆 **最佳模型保存** — 自动保存最优模型文件，带时间戳版本管理

## 安装

### 从源码安装

```bash
git clone https://github.com/anthropic/autoforge.git
cd autoforge
pip install -e ".[dev]"
```

### 额外依赖（按需安装）

```bash
# LightGBM 股票示例
pip install lightgbm scikit-learn pandas yfinance

# AI Agent 优化（Level 1 & Level 2）
pip install openai

# Dashboard 前端构建
cd dashboard && npm install && npm run build && cd ..
```

## 快速开始

### 端到端体验（5 分钟跑通）

```bash
# 1. 安装
git clone https://github.com/anthropic/autoforge.git && cd autoforge
pip install -e .
pip install lightgbm scikit-learn pandas yfinance

# 2. 运行 LightGBM 股票调优（30 轮贝叶斯优化）
mltune lgbm-stock --n-trials 30 --study-name lgbm_stock_price

# 3. 构建并启动 Dashboard
cd dashboard && npm install && npm run build && cd ..
mltune dashboard --port 8000

# 4. 打开浏览器
#    实验列表: http://localhost:8000/dashboard/#/
#    调优详情: http://localhost:8000/dashboard/#/studies/lgbm_stock_price
```

### Python API

```python
from mltune import Tuner, Config

# 定义配置
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

# 定义目标函数
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    # ... 训练你的模型 ...
    return val_loss

# 运行优化
tuner = Tuner(config)
study = tuner.optimize(objective, n_trials=50)

print(f"最佳参数: {study.best_params}")
print(f"最佳指标: {study.best_value}")
print(f"最佳模型: {study.best_model_path}")
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `mltune lgbm-stock` | LightGBM 股票价格预测调优示例 |
| `mltune agent-tune` | LLM Agent 驱动的超参搜索 |
| `mltune autoresearch` | autoresearch 风格的自主代码修改循环 |
| `mltune dashboard` | 启动 Web Dashboard |
| `mltune experiments` | 查看实验列表 |
| `mltune report <study.json>` | 从 Study 文件生成报告 |

### lgbm-stock — LightGBM 调优示例

```bash
# 使用贝叶斯优化 (TPE) 搜索 LightGBM 超参数
# 数据来源优先级: 本地缓存 (data/AAPL.csv) → yfinance 下载 → 真实合成数据 (GBM)
mltune lgbm-stock --n-trials 30 --study-name lgbm_stock_price
```

搜索空间包含 7 个超参数：`num_leaves`、`learning_rate`、`max_depth`、`feature_fraction`、`bagging_fraction`、`bagging_freq`、`min_data_in_leaf`。

完成后自动：
- 保存 Study 到 `studies/lgbm_stock_price.json`
- 记录实验到 SQLite (`mltune.db`)
- 保存最佳模型到 `models/lgbm_stock_price_best_<timestamp>.txt`
- 可在 Dashboard 实时查看调优过程

### agent-tune — LLM 驱动超参搜索（Level 1）

用大语言模型代替 TPE 作为超参建议策略。LLM 分析历史 trial 结果，推理出下一组超参。

**默认使用火山引擎 Ark（`ark-code-latest`），无需额外配置即可运行：**

```bash
# 直接运行（使用内置 Ark 端点）
mltune agent-tune --n-trials 20

# 使用本地 Ollama（覆盖默认端点）
mltune agent-tune --n-trials 20 --model llama3 --base-url http://localhost:11434/v1

# 使用其他 OpenAI 兼容 API（DeepSeek、vLLM 等）
mltune agent-tune --n-trials 20 --model deepseek-chat --base-url https://api.deepseek.com/v1
```

也可通过环境变量覆盖默认 LLM 配置：

```bash
export ANTHROPIC_AUTH_TOKEN="your-api-key"
export ANTHROPIC_BASE_URL="https://your-endpoint/v1"
export ANTHROPIC_MODEL="your-model"
```

### autoresearch — 自主代码修改循环（Level 2）

灵感来自 [karpathy/autoresearch](https://github.com/karpathy/autoresearch)。AI Agent 直接修改训练脚本：

```bash
# 使用默认 Ark 端点
mltune autoresearch \
    --train-script train.py \
    --program-md program.md \
    --metric val_loss \
    --direction minimize \
    --max-iters 50 \
    --time-budget 300

# 或指定其他模型
mltune autoresearch -t train.py -p program.md --model gpt-4o --base-url https://api.openai.com/v1
```

Agent 循环：
1. 读取 `program.md`（研究指令）+ 实验历史
2. 提出 **一个** 有针对性的代码修改
3. 执行训练（固定时间预算）
4. 指标改善 → 保留修改；否则 → 回滚代码
5. 记录到 Study / Dashboard，保存最佳脚本到 `models/<study_name>_best_<timestamp>.py`

## 优化策略

| 策略 | Config 值 | 实现方式 | 适用场景 |
|------|-----------|---------|---------|
| **贝叶斯 (TPE)** | `bayesian` / `tpe` | Optuna TPESampler | 通用首选，高效 |
| **随机搜索** | `random` | 均匀随机采样 | 基线对比，高维空间 |
| **网格搜索** | `grid` | 穷举所有组合 | 参数少、离散空间 |
| **AI Agent** | `agent` | LLM 推理建议超参 | 利用 LLM 语义理解 |

通过 `Config` 的 `tuning.strategy` 字段切换：

```python
config = Config.from_dict({
    "tuning": {"strategy": "agent", "n_trials": 20, ...}
})
tuner = Tuner(config)
study = tuner.optimize(objective)
```

## Web Dashboard

Dashboard 基于 **React + Vite** 构建，使用暗色赛博朋克极客风主题，由 FastAPI 后端提供数据 API 和静态文件服务。

### 界面特性

- **暗色极客主题** — 深色背景 + 霓虹青/绿色调 + 网格线底纹 + 辉光效果
- **实验列表** — 查看所有实验的 ID、名称、状态（发光胶囊标签）、最佳指标、模型路径
- **Study 详情面板** — 实时状态指示器（运行/完成）、进度条、指标卡片
- **优化历史图表** — 渐变填充折线图，霓虹辉光效果，鼠标悬停显示 Tooltip
- **试验详情表** — 所有 Trial 的状态、值、耗时、参数，最优 Trial 高亮标记 ★
- **最佳模型路径** — 展示保存的最优模型文件位置
- **参数重要性** — 青→绿渐变条形图，展示各超参对目标的影响
- **实时轮询** — 每 3 秒自动刷新，可手动暂停/恢复
- **自适应精度** — Y 轴标签和 Tooltip 根据数值范围自动调整小数位

### 构建与启动

```bash
# 构建前端（仅需一次，代码变更后重新构建）
cd dashboard
npm install
npm run build
cd ..

# 启动 Dashboard（必须在项目根目录运行）
mltune dashboard --port 8000
```

打开浏览器访问 `http://localhost:8000/dashboard/#/`

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/experiments` | 实验列表（含最佳模型路径） |
| GET | `/experiments/{id}` | 实验详情 |
| GET | `/experiments/{id}/metrics` | 实验指标数据 |
| GET | `/studies/{name}` | Study 结果（含 history、best_model_path） |
| GET | `/studies/{name}/importance` | 参数重要性 |
| GET | `/studies/{name}/trials` | 所有 Trial 详情（状态、值、耗时、参数） |

## 实时监控

AutoForge 支持训练过程的实时监控。优化器在每轮 Trial 完成后增量保存 Study JSON，Dashboard 通过轮询自动获取最新数据：

```
优化进行中...
  ┌──────────────────────────────────────────────────┐
  │ ● RUNNING    12 completed / 0 failed / 30 total  │
  │ ████████████░░░░░░░░░░░░  40%                    │
  │                                                  │
  │ Best Value: 1.0001328    Completed: 12  Failed: 0│
  └──────────────────────────────────────────────────┘
```

增量保存机制：
- `Tuner.optimize()` — 每轮 Trial 后保存到 `studies/<name>.json`
- `AutoResearchRunner.run()` — 每轮迭代后保存
- Dashboard 每 3 秒轮询 `/studies/{name}` 和 `/studies/{name}/trials`

## 模型保存

调优完成后，AutoForge 自动保存最佳模型（带时间戳防覆盖）：

| 命令 | 保存格式 | 路径示例 |
|------|---------|---------|
| `lgbm-stock` | `.txt` + `.pkl` | `models/lgbm_stock_price_best_20260314_103000.txt` |
| `agent-tune` | `.txt` + `.pkl` | `models/agent_lgbm_best_20260314_103000.txt` |
| `autoresearch` | `.py` (最佳脚本) | `models/autoresearch_best_20260314_103000.py` |

模型路径会记录到 Study 中，并在 Dashboard 的实验列表和 Study 详情页展示。

## 项目结构

```
autoforge/
├── mltune/                     # Python 包（保留包名兼容性）
│   ├── __init__.py
│   ├── cli.py                  # CLI 入口（所有命令）
│   ├── core/                   # 核心抽象
│   │   ├── config.py           # 配置管理（YAML / Dict）
│   │   ├── experiment.py       # 实验生命周期
│   │   └── registry.py         # 组件注册
│   ├── optim/                  # 优化引擎
│   │   ├── base.py             # BaseOptimizer + Trial 接口
│   │   ├── bayesian.py         # 贝叶斯优化（Optuna TPE）
│   │   ├── grid.py             # 网格搜索 + 随机搜索
│   │   ├── agent.py            # AgentOptimizer + AutoResearchRunner
│   │   ├── study.py            # Study（试验结果集合 + 增量保存）
│   │   └── tuner.py            # Tuner 高层接口
│   ├── tracker/                # 实验追踪
│   │   ├── backend.py          # SQLite 存储后端
│   │   ├── metrics.py          # 指标采集
│   │   └── visualizer.py       # 可视化工具
│   ├── api/                    # Web API
│   │   └── routes.py           # FastAPI 路由 + 静态文件挂载
│   └── utils/                  # 工具函数
│       ├── common.py
│       ├── device.py
│       └── seed.py
├── dashboard/                  # Web Dashboard（React + Vite）
│   ├── src/
│   │   ├── App.tsx             # 路由与布局
│   │   ├── api.ts              # 后端 API 调用
│   │   ├── styles.css          # 暗色赛博朋克主题样式
│   │   └── pages/
│   │       ├── ExperimentsList.tsx  # 实验列表（发光状态标签）
│   │       ├── ExperimentDetail.tsx # 实验详情 + 指标图表
│   │       └── StudyView.tsx       # 调优详情（实时监控 + 图表）
│   ├── vite.config.ts
│   └── package.json
├── examples/                   # 示例脚本
│   ├── simple_optimization.py  # 基础优化示例
│   └── agent_optimization.py   # AI Agent 优化示例
├── train.py                    # AutoResearch 示例训练脚本
├── program.md                  # AutoResearch 研究指令
├── configs/
│   └── example_config.yaml     # 配置模板
├── data/                       # 缓存数据（如 AAPL.csv）
├── models/                     # 最佳模型保存目录（带时间戳）
├── studies/                    # Study JSON 输出（增量保存）
├── logs/                       # 日志
├── tests/                      # 测试
├── pyproject.toml              # 项目依赖与构建配置
├── README.md                   # English documentation
└── README_zh.md                # 中文文档
```

## 数据流

```
                    ┌──────────────┐
                    │  Config      │ YAML / Dict 配置
                    │  (strategy,  │
                    │  search_space)│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Tuner      │ 高层接口，选择优化器
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
                    │  objective() │ 用户定义的训练函数
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Study      │ 收集所有 Trial 结果
                    │  (增量保存)    │ 每轮 Trial 后写入 JSON
                    └──────┬───────┘
                           │
         ┌─────────┬───────┼───────┬─────────┐
         │         │       │       │         │
    ┌────▼──┐ ┌────▼──┐ ┌──▼───┐ ┌▼──────┐ ┌▼─────────┐
    │ JSON  │ │SQLite │ │Models│ │ API   │ │Dashboard │
    │(study)│ │(.db)  │ │(.txt │ │(Fast  │ │(React)   │
    │       │ │       │ │.pkl  │ │ API)  │ │暗色极客风  │
    │       │ │       │ │.py)  │ │       │ │实时轮询    │
    └───────┘ └───────┘ └──────┘ └───────┘ └──────────┘
```

## 高级用法

### 自定义优化器

```python
from mltune.optim import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def suggest(self, trial):
        """为当前 trial 建议参数。"""
        return {"lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True)}

    def tell(self, trial, value):
        """接收 trial 结果。"""
        pass

optimizer = MyOptimizer(config)
study = optimizer.optimize(objective, n_trials=100)
```

### AgentOptimizer Python API

```python
from mltune.optim.agent import AgentOptimizer

# 默认使用 Ark（ark-code-latest），无需传任何 LLM 参数
optimizer = AgentOptimizer(config, temperature=0.7, fallback_to_random=True)
study = optimizer.optimize(objective, n_trials=20)

# 覆盖为其他端点
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

# 默认使用 Ark（ark-code-latest）
runner = AutoResearchRunner(
    train_script="train.py",
    program_md="program.md",
    eval_metric_name="val_loss",
    direction="minimize",
)
study = runner.run(max_iterations=50, time_budget_per_run=300)

# 最佳脚本保存到 models/autoresearch_best_<timestamp>.py
print(f"Best model: {study.best_model_path}")
```

## 开发

```bash
# 克隆并安装
git clone https://github.com/anthropic/autoforge.git
cd autoforge
pip install -e ".[dev]"

# 运行测试
pytest

# 格式化代码
black mltune tests
isort mltune tests

# 类型检查
mypy mltune
```

### Dashboard 开发模式

```bash
cd dashboard
npm install
npm run dev    # 开发服务器 http://localhost:5173
npm run build  # 生产构建到 dist/
```

Dashboard 技术栈：
- **React 18** + **TypeScript** — 前端框架
- **Vite** — 构建工具（`base: "/dashboard/"` 配合 FastAPI 静态服务）
- **HashRouter** — 客户端路由，无需服务端路由配置
- **SVG** — 图表渲染（优化历史折线图、参数重要性条形图）
- **Polling** — 每 3 秒轮询后端 API 实现实时更新

## 致谢

- 灵感来自 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch)
- 贝叶斯优化基于 [Optuna](https://optuna.org/)
- Dashboard UI 参考 [TensorBoard](https://www.tensorflow.org/tensorboard) 和 [W&B](https://wandb.ai/)

## 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**[示例](examples/)** •
**[Changelog](CHANGELOG.md)**

</div>
