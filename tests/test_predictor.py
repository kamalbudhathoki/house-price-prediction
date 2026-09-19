import numpy as np
import pandas as pd
import pytest

from predictor import HousePricePredictor
from preprocessing import build_pipeline
from save_model import build_defaults

DATA_PATH = "data/train.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def build_small_predictor(df, constant_log=12.5):
    X = df.drop(columns=["SalePrice"]).head(100)
    pipeline = build_pipeline().fit(X)

    class ConstantModel:
        def predict(self, X):
            return np.full(X.shape[0], constant_log)

    defaults = build_defaults(X)

    return HousePricePredictor(
        pipeline, ConstantModel(), defaults, X.columns.tolist()
    )


class TestHousePricePredictor:
    def test_returns_positive_float(self, df):
        predictor = build_small_predictor(df)
        price = predictor.predict({"OverallQual": 7, "GrLivArea": 1500})
        assert isinstance(price, float)
        assert price > 0
        assert np.isclose(price, float(np.expm1(12.5)))

    def test_prediction_with_empty_inputs_uses_defaults(self, df):
        predictor = build_small_predictor(df)
        price = predictor.predict({})
        assert np.isclose(price, float(np.expm1(12.5)))

    def test_defaults_fill_unknown_columns(self, df):
        predictor = build_small_predictor(df)
        row = {col: predictor.defaults.get(col) for col in predictor.columns}
        predictor.predict({})  # must not raise despite missing many columns
        assert all(col in predictor.defaults for col in predictor.columns)

    def test_inputs_override_defaults(self, df):
        X = df.drop(columns=["SalePrice"]).head(50)
        pipeline = build_pipeline().fit(X)
        defaults = build_defaults(X)

        row_a = pd.DataFrame([dict(defaults)])
        row_b = pd.DataFrame([dict(defaults, OverallQual=9, GrLivArea=2500)])
        enc_a = pipeline.transform(row_a)
        enc_b = pipeline.transform(row_b)
        assert not np.allclose(enc_a, enc_b)

        ref = pipeline.transform(pd.concat([row_a, row_b]))
        assert np.allclose(ref[0], enc_a[0])
        assert np.allclose(ref[1], enc_b[0])

    def test_scaling_applied_before_model(self, df):
        X = df.drop(columns=["SalePrice"]).head(50)
        pipeline = build_pipeline().fit(X)
        raw = pd.DataFrame([build_defaults(X)])
        encoded = pipeline.transform(raw)
        expected_width = pipeline.transform(
            pd.concat([raw, raw])).shape[1]
        assert encoded.shape == (1, expected_width)
        assert np.isfinite(encoded).all()
        assert abs(float(encoded.mean())) < 1.0