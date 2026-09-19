import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    DROPPED_COLS,
    ORDINAL_MAPS,
    FeatureEngineer,
    build_pipeline,
    prepare_data,
)

DATA_PATH = "data/train.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def sample_row(df):
    X = df.drop(columns=["SalePrice"])
    return X.head(10).copy()


class TestFeatureEngineer:
    def test_drops_high_missingness_columns(self, df):
        X = sample_row(df)
        out = FeatureEngineer().transform(X)
        for col in DROPPED_COLS:
            assert col not in out.columns

    def test_creates_engineered_features(self, df):
        X = sample_row(df)
        out = FeatureEngineer().transform(X)
        for col in ["HouseAge", "RemodAge", "TotalBath", "TotalSF", "TotalPorchSF"]:
            assert col in out.columns
        assert (out["HouseAge"] == X["YrSold"] - X["YearBuilt"]).all()
        expected_sf = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        assert np.allclose(out["TotalSF"], expected_sf)

    def test_has_pool_and_fireplace_flags(self, df):
        out = FeatureEngineer().transform(sample_row(df))
        assert out["HasPool"].isin([0, 1]).all()
        assert out["HasFireplace"].isin([0, 1]).all()
        assert (out["HasPool"] == (out["PoolArea"] > 0).astype(int)).all()

    def test_ordinal_mapping(self, df):
        X = sample_row(df)
        X["BsmtQual"] = ["Ex", "Gd", "TA", "Fa", "Po", None, "Ex", "Gd", "TA", "Fa"]
        out = FeatureEngineer().transform(X)
        mapping = ORDINAL_MAPS["BsmtQual"]
        expected = ["Ex", "Gd", "TA", "Fa", "Po", "NA", "Ex", "Gd", "TA", "Fa"]
        expected = [mapping[v] for v in expected]
        assert out["BsmtQual"].tolist() == expected

    def test_no_missing_values_after_transform(self, df):
        out = FeatureEngineer().transform(sample_row(df))
        numeric_na = out.select_dtypes(include=np.number).isna().sum().sum()
        assert numeric_na == 0
        ohe_pending = {c for c in out.columns if out[c].isna().any()}
        assert ohe_pending.issubset({"MasVnrType", "GarageType"})

    def test_lotfrontage_imputed(self, df):
        X = sample_row(df).copy()
        X.loc[X.index[0], "LotFrontage"] = np.nan
        out = FeatureEngineer().transform(X)
        assert not pd.isna(out["LotFrontage"]).any()


class TestPreparedData:
    def test_split_ratio_and_shape(self, df):
        result = prepare_data(df)
        n = len(df)
        assert result["X_train"].shape[0] == int(round(n * 0.8))
        assert result["X_test"].shape[0] == n - result["X_train"].shape[0]
        assert result["X_train"].shape[1] == result["X_test"].shape[1]

    def test_no_nans_in_encoded_output(self, df):
        result = prepare_data(df)
        assert not np.isnan(result["X_train"]).any()
        assert not np.isnan(result["X_test"]).any()

    def test_feature_names_match_width(self, df):
        result = prepare_data(df)
        assert len(result["feature_names"]) == result["X_train"].shape[1]

    def test_log_target_transform(self, df):
        result = prepare_data(df)
        assert np.allclose(np.expm1(result["y_train"]), result["y_train_raw"])
        assert np.allclose(np.log1p(result["y_train_raw"]), result["y_train"])
        assert (result["y_train"] > 0).all()

    def test_reproducible_split(self, df):
        r1 = prepare_data(df, random_state=42)
        r2 = prepare_data(df, random_state=42)
        assert np.allclose(r1["y_train_raw"], r2["y_train_raw"])
        assert np.allclose(r1["X_train"], r2["X_train"])

    def test_different_seed_gives_different_split(self, df):
        r1 = prepare_data(df, random_state=1)
        r2 = prepare_data(df, random_state=2)
        assert not np.allclose(r1["y_train_raw"], r2["y_train_raw"])

    def test_missing_target_raises(self, df):
        with pytest.raises(ValueError):
            prepare_data(df.drop(columns=["SalePrice"]))


class TestPipeline:
    def test_pipeline_is_fittable(self, df):
        X = df.drop(columns=["SalePrice"])
        pipeline = build_pipeline()
        X_enc = pipeline.fit_transform(X.head(50))
        assert X_enc.shape[0] == 50
        assert X_enc.shape[1] > 10

    def test_pipeline_predicts_after_transform(self, df):
        X = df.drop(columns=["SalePrice"])
        pipeline = build_pipeline().fit(X.head(100))
        X_enc = pipeline.transform(X.head(3))
        assert X_enc.shape[0] == 3