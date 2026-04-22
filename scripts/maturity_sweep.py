import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.market_data import get_latest_close, estimate_hist_vol_annualised
from src.analytics.black_scholes import bs_price_european
from src.engines.monte_carlo import mc_price_european

def main():
    os.makedirs("figures", exist_ok=True)

    ticker = "JOBY"  # change to ACHR to compare
    S0, asof = get_latest_close(ticker)
    sigma, _ = estimate_hist_vol_annualised(ticker, lookback="1y")

    r = 0.03
    option_type = "call"
    K = S0  # ATM

    days_list = np.array([7, 30, 90, 180])
    T_list = days_list / 365.0

    bs_prices = []
    mc_prices = []
    mc_ci = []

    n_paths = 150_000
    seed0 = 100

    for i, T in enumerate(T_list):
        bs = bs_price_european(S0, K, float(T), r, sigma, option_type)
        mc, se = mc_price_european(S0, K, float(T), r, sigma, option_type, n_paths=n_paths, seed=seed0+i)
        bs_prices.append(bs)
        mc_prices.append(mc)
        mc_ci.append(1.96 * se)

    plt.figure()
    plt.plot(days_list, bs_prices, marker="o", label="Black–Scholes")
    plt.errorbar(days_list, mc_prices, yerr=mc_ci, fmt="o", capsize=5, label="Monte Carlo (±95% CI)")
    plt.xlabel("Days to maturity")
    plt.ylabel("ATM option price")
    plt.title(f"{ticker} ATM {option_type.upper()} price vs maturity (as of {asof})\n"
              f"S0={S0:.2f}, sigma(1y hist)={sigma:.2%}, r={r:.2%}")
    plt.legend()
    plt.tight_layout()

    out = f"figures/maturity_sweep_{ticker}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

    for d, bs, mc in zip(days_list, bs_prices, mc_prices):
        print(f"{ticker} {d:>3}D: BS={bs:.4f}, MC={mc:.4f}")

if __name__ == "__main__":
    main()
