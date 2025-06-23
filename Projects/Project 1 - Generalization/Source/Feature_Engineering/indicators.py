import pandas as pd

def add_sma(df, windows=[10, 20, 50, 100]):
    for window in windows:
        df[f"SMA_{window}"] = df['Close'].rolling(window=window).mean()
    return df

def add_ema(df, spans=[10, 20, 50, 100]):
    for span in spans:
        df[f"EMA_{span}"] = df['Close'].ewm(span=span, adjust=False).mean()
    return df

def add_pct_change(df):
    df["Pct_Change"] = df['Close'].pct_change()
    return df

def add_volatility(df, window=20):
    df[f"Rolling_STD_{window}"] = df['Close'].rolling(window=window).std()
    return df

def add_bollinger_bands(df, window=20, num_std=2):
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = sma + (num_std * std)
    df['BB_Lower'] = sma - (num_std * std)
    return df