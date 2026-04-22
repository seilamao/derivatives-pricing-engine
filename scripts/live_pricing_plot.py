import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.market_data import get_latest_close, estimate_hist_vol_annualised
from src.analytics.black_scholes import bs_price_european
from src.engines.monte_carlo import mc_price_european


def main():
    os.makedirs("figures", exist_ok=True)

    ticker = "AAPL"  # change this
    S0, asof_date = get_latest_close(ticker)
    sigma, daily_vol = estimate_hist_vol_annualised(ticker, lookback="1y")

    K = S0
    days_to_expiry = 30
    T = days_to_expiry / 365.0
    r = 0.03
    option_type = "call"

    n_paths = 200_000
    seed = 42

    bs = bs_price_european(S0, K, T, r, sigma, option_type)
    mc, se = mc_price_european(S0, K, T, r, sigma, option_type, n_paths=n_paths, seed=seed)

    # 95% CI around MC
    ci_low = mc - 1.96 * se
    ci_high = mc + 1.96 * se

    labels = ["Black–Scholes", "Monte Carlo"]
    values = [bs, mc]

    plt.figure()
    plt.bar(labels, values)  # no manual colors
    # Add MC error bar only on the MC bar (index 1)
    plt.errorbar(
        x=1,
        y=mc,
        yerr=[[mc - ci_low], [ci_high - mc]],
        fmt="none",
        capsize=6,
    )

    plt.title(f"{ticker} {option_type.upper()} (ATM, {days_to_expiry}D) — as of {asof_date}")
    plt.ylabel("Option price")
    subtitle = f"S0={S0:.2f}, K={K:.2f}, r={r:.2%}, sigma(hist 1y)={sigma:.2%}, n={n_paths:,}"
    plt.xlabel(subtitle)

    plt.tight_layout()
    out = f"figures/live_snapshot_{ticker}_{option_type}.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print("Saved:", out)
    print(f"BS={bs:.6f}, MC={mc:.6f}, SE={se:.6f}, 95% CI=[{ci_low:.6f}, {ci_high:.6f}]")


if __name__ == "__main__":
    main()
