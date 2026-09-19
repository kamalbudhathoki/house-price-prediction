# House Price Prediction

End-to-end machine-learning project that predicts residential sale prices from the
Kaggle **House Prices - Advanced Regression Techniques** dataset (Ames, Iowa, 2006–2010).

## Problem statement

Given a house described by 79 features (lot size, living area, number of rooms, quality
ratings, basement/garage attributes, neighborhood, sale conditions, etc.), predict its final
**sale price** — a continuous regression target — as accurately as possible. Success is measured
with **RMSE / MAE / R²** on a held-out test set plus **5-fold cross-validation**.

## Dataset

- `data/train.csv` — 1,460 houses × 81 columns (79 features + `Id` + `SalePrice` target)
- Target: `SalePrice`, dollars, heavily right-skewed (skew 1.88 → 0.12 after log1p)
- 19 columns contain missing values; many missings are *informative* (e.g. `PoolQC`=NaN means
  "no pool", `BsmtQual`=NaN means "no basement")

## Approach

1. **EDA** (`eda_housing.py`) — target distribution, correlation heatmap, top-5 feature
   relationships, IQR outlier analysis, and a per-column missing-value handling plan.
2. **Preprocessing** (`preprocessing.py`) — a reusable `sklearn.Pipeline`:
   - drop uninformative / >80%-missing columns (`Id`, `Utilities`, `PoolQC`, `MiscFeature`,
     `Alley`, `Fence`)
   - impute by domain semantics: `LotFrontage` → median per *Neighborhood*; `MasVnrArea` and
     `GarageYrBlt` → 0 (absence); quality columns → "NA" placeholder; safety-net `SimpleImputer`
   - ordinal encoding for quality/condition/exposure/finish/slope columns
   - one-hot encoding for 23 nominal columns
   - feature engineering: `HouseAge`, `RemodAge`, `TotalBath`, `TotalSF`, `TotalPorchSF`,
     `HasPool`, `HasFireplace`
   - `StandardScaler` on numeric/ordinal features; 80/20 split (`random_state=42`), log1p target
3. **Model comparison** (`train_compare_models.py`) — Linear, Ridge, Lasso, Random Forest,
   XGBoost, each evaluated with 5-fold CV (raw-scale RMSE) + test RMSE/MAE/R².
4. **Tuning** (`tune_xgboost.py`) — `RandomizedSearchCV` (40 draws × 5 folds) over the CV
   winner (XGBoost).
5. **Deployment artifact** (`save_model.py`) — retrains the tuned model inside the full
   preprocessing pipeline, bundles pipeline + model + per-column defaults, and saves the
   artifact with `joblib`; served by a Streamlit app (`app.py`).

## Model performance

| Model | CV RMSE (5-fold) | CV RMSE std | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| Linear Regression | $45,972 | $28,888 | $22,731 | $15,054 | 0.9326 |
| Ridge (CV alpha) | $45,692 | $28,457 | $22,739 | $15,028 | 0.9326 |
| Lasso (CV alpha) | $49,570 | $38,456 | $22,890 | $14,460 | 0.9317 |
| XGBoost | $28,046 | $5,489 | $25,326 | $15,393 | 0.9164 |
| Random Forest | $29,684 | $4,819 | $29,879 | $17,129 | 0.8836 |
| **XGBoost (tuned)** | **$25,386** | — | $26,889 | $15,366 | 0.9057 |

**Winner:** XGBoost won the fair cross-validated comparison (lowest, most stable CV RMSE).
Tuning (`learning_rate=0.05, max_depth=3, n_estimators=300, subsample=0.9,
colsample_bytree=0.7`) improved CV RMSE by ~9.5%. The linear models show excellent single-split
test error but unstable CV — splits containing expensive outlier homes destabilize them.

## Project structure

```
house-price-prediction/
├── data/
│   ├── train.csv               # Kaggle training set (1460 × 81)
│   └── processed/              # encoded+scaled train/test splits
├── models/
│   └── house_price_model.joblib  # saved pipeline + model artifact
├── plots/                      # EDA figures
├── results/                    # comparison tables, tuning charts
├── eda_housing.py              # exploratory data analysis
├── preprocessing.py            # pipeline builder + train/test split
├── train_compare_models.py     # 5-model comparison
├── tune_xgboost.py             # hyperparameter tuning
├── save_model.py               # builds/saves the deployment artifact
├── predictor.py                # HousePricePredictor wrapper
├── app.py                      # Streamlit app
├── tests/                      # unit tests (preprocessing & prediction)
├── requirements.txt
└── README.md
```

## Setup and run tests

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## Run the Streamlit app locally

1. Make sure the model artifact exists: `models/house_price_model.joblib`.
   Create it if missing with `python save_model.py`.
2. Launch the app:

```bash
streamlit run app.py
```

3. Open the local URL shown in the terminal (default: http://localhost:8501).
4. Fill in the house features and click **Predict sale price**.

The app loads the artifact once (cached), fills any feature you do not set with training
median/mode defaults, runs the preprocessing pipeline + tuned XGBoost, and returns the
predicted price in dollars.

## Reproduce the pipeline end-to-end

```bash
python eda_housing.py            # EDA -> plots/
python preprocessing.py          # split + encode + scale -> data/processed/
python train_compare_models.py   # 5-model comparison -> results/model_comparison.csv
python tune_xgboost.py           # hyperparameter tuning (2-3 min)
python save_model.py             # train final model -> models/house_price_model.joblib
```