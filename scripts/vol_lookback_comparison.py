import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.market_data import estimate_hist_vol_annualised

def main():
    os.makedirs("figures", exist_ok=True)

    tickers = ["ACHR", "JOBY"]
    lookbacks = ["3mo", "6mo", "1y", "2y"]

    # store vol estimates: rows=tickers, cols=lookbacks
    vols = np.zeros((len(tickers), len(lookbacks)))

    for i, tkr in enumerate(tickers):
        for j, lb in enumerate(lookbacks):
            try:
                sigma, _ = estimate_hist_vol_annualised(tkr, lookback=lb)
                vols[i, j] = sigma
            except Exception:
                vols[i, j] = np.nan

    x = np.arange(len(lookbacks))
    width = 0.35

    plt.figure()
    for i, tkr in enumerate(tickers):
        plt.bar(x + (i - 0.5) * width, vols[i], width=width, label=tkr)

    plt.xticks(x, lookbacks)
    plt.ylabel("Annualised historical volatility")
    plt.title("Volatility estimates across different lookback windows")
    plt.legend()
    plt.tight_layout()

    out = "figures/vol_lookback_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

    # print values for your report
    for i, tkr in enumerate(tickers):
        row = ", ".join(f"{lb}={vols[i,j]:.2%}" if np.isfinite(vols[i,j]) else f"{lb}=NA"
                        for j, lb in enumerate(lookbacks))
        print(f"{tkr}: {row}")

if __name__ == "__main__":
    main()
