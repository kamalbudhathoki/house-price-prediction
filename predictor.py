import numpy as np
import pandas as pd


class HousePricePredictor:
    def __init__(self, pipeline, model, defaults, columns):
        self.pipeline = pipeline
        self.model = model
        self.defaults = defaults
        self.columns = list(columns)

    def predict(self, inputs):
        row = {col: self.defaults.get(col) for col in self.columns}
        row.update({k: v for k, v in inputs.items() if k in self.columns and v is not None})
        X = pd.DataFrame([row], columns=self.columns)
        log_price = self.model.predict(self.pipeline.transform(X))
        return float(np.expm1(log_price)[0])