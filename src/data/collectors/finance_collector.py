import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def save_ticker_data(ticker, start_date="2000-01-01", end_date="2024-01-01", output_dir="data"):
    """
    Download and save daily stock data for a single ticker from yFinance.
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL')
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format  
    output_dir (str): Directory to save CSV files
    
    Returns:
    bool: True if successful, False if failed
    """
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        # Check if data was retrieved
        if data.empty:
            print(f"No data found for ticker: {ticker}")
            return False
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Save to CSV
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath, index=False)
        
        print(f"Successfully saved {len(data)} records for {ticker} to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return False

def process_ticker_list(ticker_csv_path, start_date="2000-01-01", end_date="2024-01-01", output_dir="data"):
    """
    Process multiple tickers from a CSV file.
    
    Parameters:
    ticker_csv_path (str): Path to CSV file containing ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    output_dir (str): Directory to save CSV files
    
    Note: CSV should have a column named 'ticker' or the first column will be used
    """
    
    try:
        # Read ticker list
        tickers_df = pd.read_csv(ticker_csv_path)
        
        # Get ticker column (assume first column if 'ticker' not found)
        if 'ticker' in tickers_df.columns:
            ticker_list = tickers_df['ticker'].tolist()
        else:
            ticker_list = tickers_df.iloc[:, 0].tolist()
        
        print(f"Processing {len(ticker_list)} tickers...")
        
        success_count = 0
        failed_tickers = []
        
        # Process each ticker
        for ticker in ticker_list:
            success = save_ticker_data(ticker, start_date, end_date, output_dir)
            if success:
                success_count += 1
            else:
                failed_tickers.append(ticker)
        
        print(f"\nProcessing complete:")
        print(f"Successful: {success_count}/{len(ticker_list)}")
        if failed_tickers:
            print(f"Failed tickers: {failed_tickers}")
            
    except Exception as e:
        print(f"Error processing ticker list: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Test single ticker
    save_ticker_data("AAPL")
    
    # Example of processing multiple tickers from CSV
    # process_ticker_list("tickers.csv")