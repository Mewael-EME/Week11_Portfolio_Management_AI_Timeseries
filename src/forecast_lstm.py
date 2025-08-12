import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm(data_path: str, seq_length=60, train_end="2023-12-31", epochs=20, batch_size=32):
    """
    Train and evaluate an LSTM model for Tesla stock price forecasting.

    Parameters:
        data_path (str): Path to CSV with 'Date' and 'TSLA'.
        seq_length (int): Look-back window size.
        train_end (str): Date to split train/test.
        epochs (int): Training epochs.
        batch_size (int): Batch size.

    Returns:
        dict: Model, predictions, and evaluation metrics.
    """
    df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    tsla_prices = df['TSLA'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(tsla_prices)

    # Convert train_end to datetime
    train_end = pd.to_datetime(train_end)

    # Handle missing train_end date in index
    train_end = pd.to_datetime(train_end)
    if train_end in df.index:
    	split_idx = df.index.get_loc(train_end) + 1
    else:
    	pos = df.index.get_indexer([train_end], method='ffill')[0]
    	if pos == -1:
        	# train_end is before the first date in index, handle this case
        	raise ValueError(f"The train_end date {train_end} is before the earliest date in data.")
    	split_idx = pos + 1


    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx-seq_length:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

    plt.figure(figsize=(10, 5))
    plt.plot(df.index[split_idx:], y_test_actual, label="Actual")
    plt.plot(df.index[split_idx:], predictions, label="LSTM Forecast", color="red")
    plt.legend()
    plt.title("Tesla Stock Price Forecast (LSTM)")
    plt.show()

    return {
        "model": model,
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
