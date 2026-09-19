import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5


def load_processed():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["SalePrice"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["SalePrice"]
    return X_train, X_test, y_train, y_test


MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge (CV alpha)": RidgeCV(),
    "Lasso (CV alpha)": LassoCV(random_state=RANDOM_STATE, max_iter=100000),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        enable_categorical=False,
        tree_method="hist",
    ),
}

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def rmse_raw(y_true, y_pred):
    return root_mean_squared_error(np.expm1(y_true), np.expm1(y_pred))


raw_rmse_scorer = make_scorer(rmse_raw, greater_is_better=False)


def evaluate_models(X_train, X_test, y_train, y_test):
    y_log_train = np.log1p(y_train)
    y_log_test = np.log1p(y_test)

    rows = []
    for name, model in MODELS.items():
        pipeline = make_pipeline(StandardScaler(), model)

        cv_scores = -cross_val_score(
            pipeline, X_train, y_log_train, cv=cv, scoring=raw_rmse_scorer, n_jobs=-1
        )

        pipeline.fit(X_train, y_log_train)
        pred_log = pipeline.predict(X_test)
        pred_raw = np.expm1(pred_log)

        rmse = root_mean_squared_error(y_test, pred_raw)
        mae = mean_absolute_error(y_test, pred_raw)
        r2 = r2_score(y_test, pred_raw)

        rows.append({
            "Model": name,
            f"CV RMSE ({N_FOLDS}-fold)": cv_scores.mean(),
            "CV RMSE std": cv_scores.std(),
            "Test RMSE": rmse,
            "Test MAE": mae,
            "Test R2": r2,
        })
        print(f"  {name:20s} CV_RMSE=${cv_scores.mean():,.0f} "
              f"+/-{cv_scores.std():,.0f} | Test_RMSE=${rmse:,.0f} "
              f"MAE=${mae:,.0f} R2={r2:.4f}")

    results = (
        pd.DataFrame(rows)
        .sort_values("Test RMSE")
        .reset_index(drop=True)
    )
    return results


def main():
    X_train, X_test, y_train, y_test = load_processed()
    print(f"Data: train={X_train.shape}, test={X_test.shape} (all features pre-scaled/encoded)\n")
    results = evaluate_models(X_train, X_test, y_train, y_test)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\nComparison table:")
    print(results.to_string(index=False))

    out_path = OUT_DIR / "model_comparison.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path.resolve()}")

    best = results.iloc[0]
    print(f"\nBest model by test RMSE: {best['Model']} "
          f"(RMSE=${best['Test RMSE']:,.0f}, R2={best['Test R2']:.4f})")


if __name__ == "__main__":
    main()