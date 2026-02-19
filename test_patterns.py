"""Quick test for the updated data fetcher."""
import sys
sys.path.insert(0, ".")
import yfinance as yf
import pandas as pd

print(f"yfinance version: {yf.__version__}")

# Test yf.download directly
for symbol in ["AAPL", "BTC-USD"]:
    print(f"\n--- {symbol} ---")
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"  Rows: {len(df)}")
        if len(df) > 0:
            print(f"  Columns: {list(df.columns)}")
            last_close = df["Close"].iloc[-1]
            print(f"  Last close: {last_close:.2f}")
            print(f"  Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone!")
