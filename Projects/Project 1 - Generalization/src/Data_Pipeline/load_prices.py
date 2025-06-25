import yfinance as yf
import pandas as pd
import os
from indicators import add_sma, add_ema, add_pct_change, add_volatility, add_bollinger_bands

def download_stock_data(ticker="NVDA", start="1999-01-22", end="2019-01-22", save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    
    if data is None or data.empty:
        raise ValueError(f"No data found for {ticker}")
    
    data = data.dropna().sort_index()

    data = add_pct_change(data)
    data = add_sma(data, windows=[10, 20, 50])
    data = add_ema(data, spans=[10, 20, 50])
    data = add_volatility(data, window=20)
    data = add_bollinger_bands(data, window=20)
    
    file_path = os.path.join(save_dir, f"{ticker}_ohlcv.csv")
    data.to_csv(file_path)
    print(f"Saved to {file_path}")
    return data


if __name__ == "__main__":
    download_stock_data()