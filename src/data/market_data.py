from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def get_latest_close(ticker: str) -> tuple[float, str]:
    """
    Returns (latest_close_price, date_string).
    Uses the most recent available daily close (often yesterday if market not closed yet).
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")  # grab a few days to be safe
    if hist.empty:
        raise ValueError(f"No price data returned for ticker={ticker}")
    last_row = hist.iloc[-1]
    price = float(last_row["Close"])
    date_str = str(hist.index[-1].date())
    return price, date_str


def estimate_hist_vol_annualised(
    ticker: str,
    lookback: str = "1y",
    trading_days: int = 252,
) -> tuple[float, float]:
    """
    Historical volatility estimate from daily log returns.

    Returns (sigma_annualised, daily_vol).
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=lookback)
    if hist.empty or len(hist) < 30:
        raise ValueError(f"Not enough history returned for ticker={ticker}, period={lookback}")

    # Use Adjusted Close if available; yfinance typically includes 'Close' reliably.
    prices = hist["Close"].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    daily_vol = float(log_returns.std(ddof=1))
    sigma_annual = daily_vol * np.sqrt(trading_days)
    return sigma_annual, daily_vol

def rolling_hist_vol_annualised(ticker: str, period: str = "2y", window: int = 21, trading_days: int = 252):
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    prices = hist["Close"].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_daily = log_returns.rolling(window).std(ddof=1)
    rolling_annual = rolling_daily * np.sqrt(trading_days)
    return rolling_annual.dropna()