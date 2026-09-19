import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from preprocessing import build_pipeline
from predictor import HousePricePredictor

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/house_price_model.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.2

BEST_PARAMS = {
    "colsample_bytree": 0.7,
    "gamma": 0.0,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 1,
    "n_estimators": 300,
    "subsample": 0.9,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}


def build_defaults(X_train):
    defaults = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            defaults[col] = X_train[col].median()
        else:
            defaults[col] = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else None
    return defaults


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    X_train_enc = pipeline.fit_transform(X_train)

    model = XGBRegressor(**BEST_PARAMS)
    model.fit(X_train_enc, np.log1p(y_train))
    model.n_jobs = 1

    defaults = build_defaults(X_train)
    predictor = HousePricePredictor(pipeline, model, defaults, X_train.columns.tolist())

    pred = np.expm1(model.predict(pipeline.transform(X_test)))
    rmse = root_mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print("Final XGBoost (best params from tuning), retrained on train split:")
    print(f"  Test RMSE: ${rmse:,.0f}")
    print(f"  Test MAE : ${mae:,.0f}")
    print(f"  Test R2  : {r2:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(predictor, MODEL_PATH)
    print(f"Saved artifact: {MODEL_PATH.resolve()} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")

    sample = {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "TotalBsmtSF": 856,
        "FullBath": 2,
        "HalfBath": 1,
        "GarageCars": 2,
        "YearBuilt": 2003,
        "ExterQual": "Gd",
        "KitchenQual": "Gd",
        "Fireplaces": 1,
        "CentralAir": "Y",
    }
    price = predictor.predict(sample)
    print(f"Sanity check prediction (Sample input): ${price:,.0f}")


if __name__ == "__main__":
    main()