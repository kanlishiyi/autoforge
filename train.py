"""
LightGBM stock price prediction training script.

This script is designed to be modified by the AutoResearchRunner agent.
It loads AAPL stock data, engineers features, trains a LightGBM model,
and prints the evaluation metric in the format:  val_loss: <value>

The agent will iteratively modify this script to improve the metric.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

# ============================================================
# 1. Data Loading
# ============================================================

DATA_PATH = Path("data/AAPL.csv")

if DATA_PATH.exists():
    raw = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
else:
    # Generate synthetic data if no cached data
    np.random.seed(42)
    n = 2500
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    price = 120.0
    prices, volumes = [], []
    for _ in range(n):
        ret = np.random.normal(0.0003, 0.015)
        price *= (1 + ret)
        prices.append(price)
        volumes.append(int(np.random.lognormal(17, 0.5)))
    raw = pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates)

df = raw[["Close", "Volume"]].copy().dropna()

# ============================================================
# 2. Feature Engineering
# ============================================================

# Returns
df["ret_1"] = df["Close"].pct_change()

# Moving averages and volatility
for k in (3, 5, 10, 20):
    df[f"ma_{k}"] = df["Close"].rolling(k).mean()
    df[f"std_{k}"] = df["Close"].rolling(k).std()
    df[f"ret_{k}"] = df["Close"].pct_change(k)

# RSI (14-period)
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / (loss + 1e-10)
df["rsi_14"] = 100 - (100 / (1 + rs))

# Volume change
df["vol_change"] = df["Volume"].pct_change()

# Log-volume (new)
df["log_volume"] = np.log1p(df["Volume"])

# --- NEW FEATURES ---

# MACD (12/26/9)
ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
df["macd"] = ema_12 - ema_26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

# Bollinger Bands (20-period)
bb_ma = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()
df["bb_upper"] = bb_ma + 2 * bb_std
df["bb_lower"] = bb_ma - 2 * bb_std
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_ma
df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

# ATR (Average True Range) - 14 period
high = df["Close"].rolling(2).max()
low = df["Close"].rolling(2).min()
tr = high - low
tr[0] = df["Close"].iloc[0] - df["Close"].iloc[0]  # first value
df["atr_14"] = tr.rolling(14).mean()
df["atr_pct"] = df["atr_14"] / df["Close"]  # ATR as percentage of price

# Lagged returns (2-day, 3-day, 5-day lags)
df["ret_2_lag"] = df["Close"].pct_change(2)
df["ret_3_lag"] = df["Close"].pct_change(3)
df["ret_5_lag"] = df["Close"].pct_change(5)

# Lagged volume change
df["vol_change_2_lag"] = df["Volume"].pct_change(2)

# Target: next-day return
df["target"] = df["Close"].shift(-1) / df["Close"] - 1.0
df = df.dropna()

feature_cols = [
    "Close", "Volume", "log_volume", "ret_1",
    "ma_3", "ma_5", "ma_10", "ma_20",
    "std_3", "std_5", "std_10", "std_20",
    "ret_3", "ret_5", "ret_10", "ret_20",
    "rsi_14", "vol_change", "vol_change_2_lag",
    # New technical indicators
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr_14", "atr_pct",
    # Lagged returns
    "ret_2_lag", "ret_3_lag", "ret_5_lag",
]

X = df[feature_cols].astype(np.float32).values
y = df["target"].astype(np.float32).values

# ============================================================
# 3. Train/Validation Split
# ============================================================

# Changed from 0.8 to 0.85 to give model more training data
split_idx = int(len(X) * 0.85)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# --- REMOVE OUTLIERS FROM TRAINING DATA ---
# Clip returns beyond 3 standard deviations to reduce impact of extreme values on RMSE
train_mean = np.mean(y_train)
train_std = np.std(y_train)
lower_bound = train_mean - 3 * train_std
upper_bound = train_mean + 3 * train_std

# Create mask for non-outlier samples
outlier_mask = (y_train >= lower_bound) & (y_train <= upper_bound)
X_train_clean = X_train[outlier_mask]
y_train_clean = y_train[outlier_mask]

print(f"Training samples after outlier removal: {len(y_train_clean)} / {len(y_train)} "
      f"({100 * len(y_train_clean) / len(y_train):.1f}%)")

# --- CREATE SAMPLE WEIGHTS ---
# Give more weight to recent samples to help model adapt to recent market patterns
# Use exponential decay: recent samples get weight ~1.0, older samples get progressively lower weight
n_train = len(y_train_clean)
decay_rate = 0.0005  # Controls how fast weights decay for older samples
sample_weights = np.exp(decay_rate * np.arange(n_train))
# Normalize weights to have mean = 1
sample_weights = sample_weights / sample_weights.mean()

train_set = lgb.Dataset(X_train_clean, label=y_train_clean, weight=sample_weights)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

# ============================================================
# 4. Model Training — Agent can modify these hyperparameters
# ============================================================

# Using L1 loss (MAE) which is more robust to outliers in stock returns
params = {
    "objective": "regression_l1",  # Changed from regression (L2) to L1 (MAE) for robustness
    "metric": "rmse",  # Still evaluate using RMSE for comparison with previous runs
    "verbosity": -1,
    "boosting_type": "gbdt",
    "feature_pre_filter": False,
    # --- Hyperparameters: lower learning rate with more rounds for finer convergence ---
    "num_leaves": 31,
    "learning_rate": 0.01,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "min_data_in_leaf": 30,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
}

NUM_BOOST_ROUND = 1500
EARLY_STOPPING_ROUNDS = 80

callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]

model = lgb.train(
    params,
    train_set,
    num_boost_round=NUM_BOOST_ROUND,
    valid_sets=[val_set],
    callbacks=callbacks,
)

# ============================================================
# 5. Evaluation — print metric for AutoResearchRunner to parse
# ============================================================

y_pred = model.predict(X_val)
diff = y_pred - y_val
rmse = float(np.sqrt(np.mean(diff * diff)))

# >>> This line is parsed by AutoResearchRunner <<<
print(f"val_loss: {rmse}")
