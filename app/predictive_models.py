import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ARIMAModel:
    """
    Wrapper for ARIMA time series forecasting.
    """
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.metrics = {}

    def fit(self, series: pd.Series):
        self.model = ARIMA(series, order=self.order)
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
        return self.fitted_model.forecast(steps=steps)

    def get_metrics(self):
        return self.metrics

class SARIMAModel:
    """
    Wrapper for SARIMA (Seasonal ARIMA) time series forecasting.
    """
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.metrics = {}

    def fit(self, series: pd.Series):
        self.model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)
        
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
        return self.fitted_model.forecast(steps=steps)

    def get_metrics(self):
        return self.metrics

# Manual grid search for ARIMA

def grid_search_arima(series, p_values=[0,1,2], d_values=[0,1], q_values=[0,1,2]):
    best_score, best_cfg, best_model = float("inf"), None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(series, order=(p,d,q))
                    model_fit = model.fit()
                    y_pred = model_fit.fittedvalues
                    rmse = np.sqrt(np.mean((series - y_pred) ** 2))
                    if rmse < best_score:
                        best_score, best_cfg, best_model = rmse, (p,d,q), model_fit
                except Exception:
                    continue
    return best_model, best_cfg, best_score

# Manual grid search for SARIMA

def grid_search_sarima(series, p_values=[0,1], d_values=[0,1], q_values=[0,1], P_values=[0,1], D_values=[0,1], Q_values=[0,1], m=12):
    best_score, best_cfg, best_model = float("inf"), None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                model = SARIMAX(series, order=(p,d,q), seasonal_order=(P,D,Q,m))
                                model_fit = model.fit(disp=False)
                                y_pred = model_fit.fittedvalues
                                rmse = np.sqrt(np.mean((series - y_pred) ** 2))
                                if rmse < best_score:
                                    best_score, best_cfg, best_model = rmse, (p,d,q,P,D,Q,m), model_fit
                            except Exception:
                                continue
    return best_model, best_cfg, best_score

# Example usage (uncomment to test):
# data = pd.Series(np.random.randn(100))
# arima = ARIMAModel(order=(2,1,2))
# arima.fit(data)
# print(arima.forecast(5))
# sarima = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,12))
# sarima.fit(data)
# print(sarima.forecast(5)) 