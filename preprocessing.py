from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "SalePrice"
DATA_PATH = Path("data/train.csv")
OUT_DIR = Path("data/processed")

DROPPED_COLS = ["Id", "Utilities", "PoolQC", "MiscFeature", "Alley", "Fence"]

ORDINAL_MAPS = {
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageFinish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LandSlope": {"Gtl": 1, "Mod": 2, "Sev": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_maps=None, dropped_cols=None):
        self.ordinal_maps = ordinal_maps if ordinal_maps is not None else ORDINAL_MAPS
        self.dropped_cols = dropped_cols if dropped_cols is not None else DROPPED_COLS

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in self.dropped_cols if c in X.columns])

        X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        X["LotFrontage"] = X["LotFrontage"].fillna(X["LotFrontage"].median())
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)
        X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])

        for col, mapping in self.ordinal_maps.items():
            if col in X.columns:
                X[col] = X[col].fillna("NA").map(mapping)

        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["TotalBath"] = (
            X["FullBath"] + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
        )
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        return X


def build_pipeline():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, make_column_selector(dtype_exclude=np.number)),
        ]
    )
    return Pipeline([
        ("engineer", FeatureEngineer()),
        ("preprocessor", preprocessor),
    ])


def prepare_data(df, target=TARGET, test_size=0.2, random_state=42, log_target=True):
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline()
    X_train_enc = pipeline.fit_transform(X_train)
    X_test_enc = pipeline.transform(X_test)

    y_train_enc = np.log1p(y_train) if log_target else y_train
    y_test_enc = np.log1p(y_test) if log_target else y_test

    engineered = pipeline.named_steps["engineer"].transform(X_train)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out(
        engineered.columns
    )

    return {
        "pipeline": pipeline,
        "X_train": X_train_enc,
        "X_test": X_test_enc,
        "y_train": y_train_enc,
        "y_test": y_test_enc,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "feature_names": feature_names,
        "log_target": log_target,
    }


def save_result(result, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    features = result["feature_names"]
    pd.DataFrame(result["X_train"], columns=features).to_csv(out_dir / "X_train.csv", index=False)
    pd.DataFrame(result["X_test"], columns=features).to_csv(out_dir / "X_test.csv", index=False)
    pd.DataFrame({"SalePrice": result["y_train_raw"]}).to_csv(out_dir / "y_train.csv", index=False)
    pd.DataFrame({"SalePrice": result["y_test_raw"]}).to_csv(out_dir / "y_test.csv", index=False)
    print(f"Saved processed datasets to: {out_dir.resolve()}")


def main():
    df = pd.read_csv(DATA_PATH)
    result = prepare_data(df)
    print(f"Split: train={result['X_train'].shape}, test={result['X_test'].shape}")
    n_num = len(result["pipeline"].named_steps["preprocessor"].transformers_[0][2])
    n_cat = len(result["pipeline"].named_steps["preprocessor"].transformers_[1][2])
    print(f"Engineered features (total after pipeline): {result['X_train'].shape[1]}")
    print(f"  numeric/ordinal/engineered -> scaled: {n_num}, categorical -> one-hot: {n_cat}")
    save_result(result)


if __name__ == "__main__":
    main()