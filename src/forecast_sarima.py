import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def train_sarima(data_path: str, train_end="2023-12-31", seasonal_period=12):
    """
    Train and evaluate a SARIMA model for Tesla stock price forecasting.

    Parameters:
        data_path (str): Path to CSV file containing 'Date' and 'TSLA' columns.
        train_end (str): Date string to split training and testing data.
        seasonal_period (int): The number of periods in a season (e.g., 12 for monthly data with yearly seasonality).

    Returns:
        dict: Contains the trained model, forecast, and evaluation metrics (MAE, RMSE, MAPE).
    """
    df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    tsla_prices = df['TSLA']

    train_end = pd.to_datetime(train_end)
    if train_end in df.index:
        split_idx = df.index.get_loc(train_end) + 1
    else:
        pos = df.index.get_indexer([train_end], method='ffill')[0]
        if pos == -1:
            raise ValueError(f"The train_end date {train_end} is before the earliest date in the data.")
        split_idx = pos + 1

    train_data = tsla_prices.iloc[:split_idx]
    test_data = tsla_prices.iloc[split_idx:]

    # Use auto_arima with seasonal=True to get seasonal order
    auto_model = auto_arima(
        train_data,
        seasonal=True,
        m=seasonal_period,
        trace=True,
        stepwise=True,
        suppress_warnings=True
    )
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=len(test_data))

    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(train_data.index, train_data, label="Train")
    plt.plot(test_data.index, test_data, label="Test", color="orange")
    plt.plot(test_data.index, forecast, label="SARIMA Forecast", color="green")
    plt.legend()
    plt.title("Tesla Stock Price Forecast (SARIMA)")
    plt.show()

    return {
        "model": model_fit,
        "forecast": forecast,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
