import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.market_data import get_latest_close, estimate_hist_vol_annualised
from src.analytics.black_scholes import bs_price_european
from src.engines.monte_carlo import mc_price_european


def main():
    os.makedirs("figures", exist_ok=True)

    tickers = ["ACHR", "JOBY"]  # add more if you want
    r = 0.03
    days_to_expiry = 30
    T = days_to_expiry / 365.0
    option_type = "call"
    n_paths = 200_000
    seed = 42

    bs_vals = []
    mc_vals = []
    mc_ci = []
    labels = []

    for i, tkr in enumerate(tickers):
        S0, asof = get_latest_close(tkr)
        sigma, _ = estimate_hist_vol_annualised(tkr, lookback="1y")
        K = S0  # ATM

        bs = bs_price_european(S0, K, T, r, sigma, option_type)
        mc, se = mc_price_european(S0, K, T, r, sigma, option_type, n_paths=n_paths, seed=seed + i)

        bs_vals.append(bs)
        mc_vals.append(mc)
        mc_ci.append(1.96 * se)
        labels.append(f"{tkr}\n(asof {asof})")

        print(f"{tkr}: S0={S0:.2f}, sigma={sigma:.2%}, BS={bs:.4f}, MC={mc:.4f} ± {1.96*se:.4f}")

    x = np.arange(len(tickers))

    plt.figure()
    plt.bar(x - 0.15, bs_vals, width=0.3, label="Black–Scholes")
    plt.bar(x + 0.15, mc_vals, width=0.3, label="Monte Carlo")
    plt.errorbar(x + 0.15, mc_vals, yerr=mc_ci, fmt="none", capsize=6)  # 95% CI on MC

    plt.xticks(x, labels)
    plt.ylabel("ATM 30D Call Price")
    plt.title("Live Pricing Snapshot: Black–Scholes vs Monte Carlo")
    plt.legend()
    plt.tight_layout()

    out = "figures/multi_ticker_snapshot.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    main()
