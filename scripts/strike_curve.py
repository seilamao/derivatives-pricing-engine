import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.market_data import get_latest_close, estimate_hist_vol_annualised
from src.analytics.black_scholes import bs_price_european
from src.engines.monte_carlo import mc_price_european


def main():
    os.makedirs("figures", exist_ok=True)

    ticker = "AAPL"
    S0, asof_date = get_latest_close(ticker)
    sigma, _ = estimate_hist_vol_annualised(ticker, lookback="1y")

    days_to_expiry = 30
    T = days_to_expiry / 365.0
    r = 0.03
    option_type = "call"

    # Strikes around spot
    strikes = np.linspace(0.8 * S0, 1.2 * S0, 9)

    bs_prices = []
    mc_prices = []
    mc_err = []

    n_paths = 100_000
    base_seed = 100

    for i, K in enumerate(strikes):
        bs = bs_price_european(S0, float(K), T, r, sigma, option_type)
        mc, se = mc_price_european(S0, float(K), T, r, sigma, option_type, n_paths=n_paths, seed=base_seed + i)
        bs_prices.append(bs)
        mc_prices.append(mc)
        mc_err.append(1.96 * se)  # show 95% error bars

    plt.figure()
    plt.plot(strikes, bs_prices, marker="o", label="Black–Scholes")
    plt.errorbar(strikes, mc_prices, yerr=mc_err, fmt="o", capsize=4, label="Monte Carlo (±95% CI)")

    plt.title(f"{ticker} {option_type.upper()} — Price vs Strike (as of {asof_date})")
    plt.xlabel("Strike K")
    plt.ylabel("Option price")
    plt.legend()
    plt.tight_layout()

    out = f"figures/strike_curve_{ticker}_{option_type}.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print("Saved:", out)


if __name__ == "__main__":
    main()
