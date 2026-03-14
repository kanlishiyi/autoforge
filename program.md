# Research Program: LightGBM Stock Price Prediction

## Objective

Minimize `val_loss` (RMSE on validation set) for next-day stock return prediction using LightGBM.

## Current Setup

- **Data**: AAPL historical stock prices (~2500 trading days)
- **Features**: Price, Volume, returns, moving averages, volatility, RSI
- **Model**: LightGBM (gradient boosting)
- **Target**: Next-day return (Close[t+1] / Close[t] - 1)
- **Split**: 80% train / 20% validation (time-series order)

## Research Directions

Try the following improvements (one at a time):

1. **Feature Engineering**
   - Add MACD (12/26/9), Bollinger Bands, ATR
   - Add day-of-week / month features
   - Add lagged returns (2-day, 3-day, 5-day lags)
   - Normalize or standardize features
   - Try log-volume instead of raw volume

2. **Hyperparameter Tuning**
   - Adjust `num_leaves` (try 15-200)
   - Adjust `learning_rate` (try 0.001-0.1)
   - Adjust `max_depth` (try 3-12 vs unlimited)
   - Tune regularization (`lambda_l1`, `lambda_l2`)
   - Adjust `min_data_in_leaf` (try 5-100)
   - Try `feature_fraction` and `bagging_fraction` combinations

3. **Training Strategy**
   - Increase/decrease `NUM_BOOST_ROUND`
   - Adjust early stopping patience
   - Try `dart` boosting type instead of `gbdt`

4. **Data Processing**
   - Remove outliers (returns > 3 std)
   - Add rolling z-score normalization
   - Try different train/val split ratios

## Constraints

- The script must remain a single `train.py` file
- The output must always include `val_loss: <float>` on stdout
- Do not add new package dependencies beyond numpy, pandas, lightgbm
- Keep total training time under 60 seconds
- Make ONE focused change per iteration
