import datetime as dt

from src.data.market_data import get_latest_close, estimate_hist_vol_annualised
from src.analytics.black_scholes import bs_price_european
from src.engines.monte_carlo import mc_price_european


def main():
    # ---- Choose a ticker ----
    ticker = "JOBY"  # change this

    # ---- Pull latest close ----
    S0, asof_date = get_latest_close(ticker)

    # ---- Estimate volatility from history ----
    sigma, daily_vol = estimate_hist_vol_annualised(ticker, lookback="1y")

    # ---- Set option contract assumptions ----
    K = S0                 # ATM strike (simple)
    days_to_expiry = 30
    T = days_to_expiry / 365.0

    # Risk-free rate (simple constant for project)
    # You can later swap this for SONIA/UK T-bill data if you want.
    r = 0.03

    option_type = "call"
    n_paths = 200_000
    seed = 42  # keep reproducible for your report

    # ---- Price ----
    bs = bs_price_european(S0, K, T, r, sigma, option_type)
    mc, se = mc_price_european(S0, K, T, r, sigma, option_type, n_paths=n_paths, seed=seed)

    # ---- Print a clean summary ----
    print(f"Ticker: {ticker}")
    print(f"As-of (latest close): {asof_date}")
    print(f"S0 (close): {S0:.4f}")
    print(f"K: {K:.4f} (ATM)")
    print(f"T: {T:.6f} years ({days_to_expiry} days)")
    print(f"r: {r:.4f}")
    print(f"sigma (hist, 1y): {sigma:.4f} (daily vol: {daily_vol:.4f})")
    print("")
    print(f"Black–Scholes {option_type}: {bs:.6f}")
    print(f"Monte Carlo  {option_type}: {mc:.6f}  (SE: {se:.6f})")
    print(f"Abs diff (MC-BS): {abs(mc - bs):.6f}")


if __name__ == "__main__":
    main()
