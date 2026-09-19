# House Price Prediction

End-to-end ML project on the Kaggle **House Prices - Advanced Regression Techniques** dataset
(Ames, Iowa). Includes EDA, a reusable sklearn preprocessing pipeline, model training/comparison,
hyperparameter tuning, a saved model artifact, and a Streamlit web app for live predictions.

## Project files

| File | Purpose |
|------|---------|
| `eda_housing.py` | EDA: target distribution, correlation heatmap, top-feature plots, outliers, missing-value plan |
| `preprocessing.py` | Feature engineering + imputation + encoding + scaling as a reusable sklearn `Pipeline`; 80/20 split |
| `train_compare_models.py` | Trains Linear, Ridge, Lasso, Random Forest, XGBoost; 5-fold CV + test RMSE/MAE/R² table |
| `tune_xgboost.py` | RandomizedSearchCV over XGBoost; retrains and evaluates the tuned model |
| `save_model.py` | Retrains final XGBoost with best params, bundles pipeline + model + defaults, saves with joblib |
| `predictor.py` | `HousePricePredictor` wrapper used by the streamlit app |
| `app.py` | Streamlit app: form → predicted price |
| `data/train.csv` | Kaggle training set (1460 × 81) |
| `models/house_price_model.joblib` | Saved pipeline + model artifact |
| `plots/`, `results/`, `data/processed/` | Generated outputs |

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Reproduce the pipeline end-to-end

```bash
python eda_housing.py            # EDA -> plots/
python preprocessing.py          # split + encode + scale -> data/processed/
python train_compare_models.py   # 5-model comparison -> results/model_comparison.csv
python tune_xgboost.py           # hyperparameter tuning (2-3 min)
python save_model.py             # train final model -> models/house_price_model.joblib
```

If you only want to run the app (artifact already committed/made), skip straight to Run.

## Run the Streamlit app locally

1. Make sure the model artifact exists: `models/house_price_model.joblib`.
   Create it with `python save_model.py` if it is missing.
2. Launch the app:

```bash
streamlit run app.py
```

3. Open the local URL shown in the terminal (default: http://localhost:8501).
4. Fill in the house features and click **Predict sale price**.

The app loads the artifact once (cached), fills any feature you do not set with training
median/mode defaults, runs the preprocessing pipeline + tuned XGBoost, and returns the
predicted price in dollars.