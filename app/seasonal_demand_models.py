import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ProphetSeasonalModel:
    """
    Wrapper for Facebook Prophet for seasonal demand forecasting.
    """
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.metrics = {}

    def fit(self, series: pd.Series):
        # Reset index and rename columns for Prophet
        df = series.reset_index()
        df.columns = ['ds', 'y']
        
        # Initialize Prophet with yearly seasonality
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # Use multiplicative for better seasonal patterns
            changepoint_prior_scale=0.05,  # Allow for some flexibility in trend changes
            seasonality_prior_scale=10.0  # Allow for strong seasonality
        )
        
        # Fit the model
        self.fitted_model = self.model.fit(df)
        
        # Calculate performance metrics
        y_pred = self.fitted_model.predict(df)['yhat']
        y_true = df['y']
        
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'accuracy': (1 - np.mean(np.abs((y_true - y_pred) / y_true))) * 100
        }
        
        return self.fitted_model

    def forecast(self, steps=12):
        if self.fitted_model is None:
            raise ValueError("Model must be fit before forecasting.")
        
        # Create future dataframe for prediction
        future = self.model.make_future_dataframe(periods=steps, freq='MS')
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return only the forecasted values
        return forecast[['ds', 'yhat']].tail(steps)

    def get_metrics(self):
        return self.metrics

# Example usage (uncomment to test):
# data = pd.Series(np.random.randn(100), index=pd.date_range('2020-01-01', periods=100, freq='MS'))
# model = ProphetSeasonalModel()
# model.fit(data)
# print(model.forecast(12)) 