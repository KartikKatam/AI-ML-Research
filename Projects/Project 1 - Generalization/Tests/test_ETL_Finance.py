from src.Data_Pipeline.ETL_Finance import fetch_ticker
import pandas as pd

def test_fetch_nvda():
    df = fetch_ticker("NVDA", "2023-01-01", "2023-01-15")
    assert not df.empty
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert df.index.min() >= pd.Timestamp("2023-01-01")