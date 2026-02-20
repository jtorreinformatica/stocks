"""
Data fetcher module — downloads OHLCV data via yfinance (with pandas_ta fallback).
"""

import yfinance as yf
import pandas as pd
import streamlit as st

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL', 'BTC-USD').
        period: Data period (e.g. '6mo', '1y', '2y').
        interval: Data interval (e.g. '1d', '1wk', '1mo').

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        and a DatetimeIndex.
    """
    df = pd.DataFrame()

    # Primary: yf.download() — most reliable in yfinance >= 1.x
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        # yf.download may return multi-level columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        pass

    # Fallback 1: Ticker.history()
    if df is None or df.empty:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
        except Exception:
            pass

    # Fallback 2: pandas_ta
    if (df is None or df.empty) and HAS_PANDAS_TA:
        try:
            df = pd.DataFrame().ta.ticker(symbol, period=period, interval=interval)
        except Exception:
            pass

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == "open":
            col_map[col] = "Open"
        elif col_lower == "high":
            col_map[col] = "High"
        elif col_lower == "low":
            col_map[col] = "Low"
        elif col_lower == "close":
            col_map[col] = "Close"
        elif col_lower == "volume":
            col_map[col] = "Volume"
    df = df.rename(columns=col_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    return df


