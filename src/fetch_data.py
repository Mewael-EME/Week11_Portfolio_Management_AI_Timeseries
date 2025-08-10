import yfinance as yf
import pandas as pd

def download_ticker(ticker, start_date, end_date):
    try:
        # Explicitly set auto_adjust=True to get adjusted close prices by default
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            print(f"Warning: No data fetched for {ticker}.")
        return df
    except Exception as e:
        print(f"Failed to get ticker '{ticker}' reason: {e}")
        return pd.DataFrame()

def main():
    start_date = "2015-07-01"
    end_date = "2025-07-31"

    tsla_df = download_ticker("TSLA", start_date, end_date)
    bnd_df = download_ticker("BND", start_date, end_date)
    spy_df = download_ticker("SPY", start_date, end_date)

    if tsla_df.empty or bnd_df.empty or spy_df.empty:
        print("Error: One or more datasets are empty. Exiting.")
        return

    # Use 'Close' column since auto_adjust=True already adjusts prices
    tsla_close = tsla_df["Close"].copy()
    tsla_close.name = "TSLA"

    bnd_close = bnd_df["Close"].copy()
    bnd_close.name = "BND"

    spy_close = spy_df["Close"].copy()
    spy_close.name = "SPY"

    adj_close = pd.concat([tsla_close, bnd_close, spy_close], axis=1)

    adj_close.to_csv("data/adj_close_prices.csv")
    print("Saved combined adjusted close prices to data/adj_close_prices.csv")

if __name__ == "__main__":
    main()
