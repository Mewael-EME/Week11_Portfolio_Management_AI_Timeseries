import pandas as pd

def load_adj_close(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def calculate_daily_returns(df):
    """
    Calculate daily returns as percentage change.
    """
    returns = df.pct_change().dropna()
    return returns

def fill_missing_values(df):
    """
    Fill missing values using forward fill method.
    """
    return df.ffill().bfill()

if __name__ == "__main__":
    adj_close_path = "data/adj_close_prices.csv"
    adj_close = load_adj_close(adj_close_path)

    # Handle missing data
    adj_close_clean = fill_missing_values(adj_close)

    # Calculate daily returns
    returns = calculate_daily_returns(adj_close_clean)

    # Save returns to CSV
    returns.to_csv("data/returns.csv")
    print("Preprocessing done. Returns saved to data/returns.csv")