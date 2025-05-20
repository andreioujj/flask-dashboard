import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HoltWintersModel:
    """
    Wrapper for Holt-Winters Exponential Smoothing (ETS) for price trend forecasting.
    """
    def __init__(self, trend='add', seasonal='add', seasonal_periods=12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
        self.metrics = {}

    def fit(self, series: pd.Series):
        self.model = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods
        )
        self.fitted_model = self.model.fit()
        
        # Calculate performance metrics
        y_pred = self.fitted_model.fittedvalues
        y_true = series
        
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'accuracy': (1 - np.mean(np.abs((y_true - y_pred) / y_true))) * 100
        }
        
        return self.fitted_model

    def forecast(self, steps=1):
        if self.fitted_model is None:
            raise ValueError("Model must be fit before forecasting.")
        return self.fitted_model.forecast(steps)

    def get_metrics(self):
        return self.metrics

# Example usage (uncomment to test):
# data = pd.Series(np.random.randn(100))
# model = HoltWintersModel(trend='add', seasonal='add', seasonal_periods=12)
# model.fit(data)
# print(model.forecast(5)) 