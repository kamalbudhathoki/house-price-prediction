import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5

PARAM_GRID = {
    "n_estimators": [200, 300, 400, 500],
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.3],
}

N_ITER = 40


def load_processed():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["SalePrice"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["SalePrice"]
    return X_train, X_test, y_train, y_test


def rmse_raw(y_true, y_pred):
    return root_mean_squared_error(np.expm1(y_true), np.expm1(y_pred))


raw_rmse_scorer = make_scorer(rmse_raw, greater_is_better=False)


def tune_xgboost(X, y_log):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base = XGBRegressor(
        objective="reg:squarederror", n_jobs=1, random_state=RANDOM_STATE, tree_method="hist"
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        scoring=raw_rmse_scorer,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )
    search.fit(X, y_log)
    return search


def plot_predicted_vs_actual(y_test, pred, model_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, pred, s=16, alpha=0.5, edgecolors="none")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="y = x (perfect)")
    ax.set_xlabel("Actual SalePrice ($)")
    ax.set_ylabel("Predicted SalePrice ($)")
    ax.set_title(f"{model_name}: predicted vs actual prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path, top_n=20):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    top = importances.tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 8))
    top.plot(kind="barh", ax=ax, color="#2c7fb8")
    ax.set_xlabel("Gain-based feature importance")
    ax.set_title(f"Top {top_n} features (XGBoost)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return importances


def main():
    X_train, X_test, y_train, y_test = load_processed()
    y_log_train = np.log1p(y_train)
    y_log_test = np.log1p(y_test)
    feature_names = X_train.columns

    print(f"Tuning XGBoost over {N_ITER} random draws x {N_FOLDS}-fold CV...")
    search = tune_xgboost(X_train, y_log_train)

    best = search.best_estimator_
    print(f"\nBest CV RMSE: ${abs(search.best_score_):,.0f}")
    print("Best parameters:")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k}: {v}")

    pred_log = best.predict(X_test)
    pred_raw = np.expm1(pred_log)
    rmse = root_mean_squared_error(y_test, pred_raw)
    mae = mean_absolute_error(y_test, pred_raw)
    r2 = r2_score(y_test, pred_raw)
    print(f"\nTuned XGBoost on test set:")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  MAE : ${mae:,.0f}")
    print(f"  R2  : {r2:.4f}")

    fig_pa = OUT_DIR / "xgb_predicted_vs_actual.png"
    plot_predicted_vs_actual(y_test, pred_raw, "Tuned XGBoost", fig_pa)
    print(f"Saved: {fig_pa.resolve()}")

    importances = plot_feature_importance(
        best, feature_names, OUT_DIR / "xgb_feature_importance.png"
    )
    print("Saved: results/xgb_feature_importance.png")
    print("\nTop 15 features by gain:")
    print(importances.sort_values(ascending=False).head(15).round(4).to_string())

    summary = pd.DataFrame({
        "Model": ["XGBoost (tuned)"],
        "Best CV RMSE": [abs(search.best_score_)],
        "Test RMSE": [rmse],
        "Test MAE": [mae],
        "Test R2": [r2],
        "Best params": [str(search.best_params_)],
    })
    summary.to_csv(OUT_DIR / "xgb_tuned_results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'xgb_tuned_results.csv'}")


if __name__ == "__main__":
    main()