"""
ETL script to download daily OHLCV from NASDAQ Data Link Sharadar
and save one Parquet file per ticker.
Run:  python -m src.data.etl_finance --tickers data/ticker_list.csv \
      --start 2000-01-01 --end 2024-12-31
"""

import argparse, os, pathlib, time
import pandas as pd
import nasdaqdatalink as ndl
from dotenv import load_dotenv
from tqdm import tqdm
from nasdaqdatalink import ApiConfig

load_dotenv()
ApiConfig.api_key = os.getenv("NASDAQ_API_KEY")  # type: ignore[attr-defined]

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = ndl.get(f"SHARADAR/DAILY", ticker=ticker,
                 start_date=start, end_date=end, paginate=True)
    df.index = pd.to_datetime(df.index)
    # keep only OHLCV
    return df.loc[:, ['open','high','low','close','volume']]

def main(args):
    tickers = pd.read_csv(args.tickers)['ticker'].tolist()
    out_dir = pathlib.Path("data/raw/finance")
    out_dir.mkdir(parents=True, exist_ok=True)

    for tic in tqdm(tickers, desc="DL"):
        out_file = out_dir / f"{tic}.parquet"
        if out_file.exists() and not args.overwrite:
            continue
        try:
            df = fetch_ticker(tic, args.start, args.end)
            df.to_parquet(out_file, engine="pyarrow", compression="snappy")
        except Exception as e:
            print(f"[WARN] {tic}: {e}")
            time.sleep(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", required=True)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())