import os
import matplotlib.pyplot as plt
from src.data.market_data import rolling_hist_vol_annualised

def main():
    os.makedirs("figures", exist_ok=True)

    tickers = ["ACHR", "JOBY"]
    window = 21  # ~1 trading month

    plt.figure()
    for tkr in tickers:
        vol = rolling_hist_vol_annualised(tkr, period="2y", window=window)
        plt.plot(vol.index, vol.values, label=tkr)

    plt.title(f"Rolling {window}-day annualised historical volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualised volatility")
    plt.legend()
    plt.tight_layout()

    out = "figures/rolling_vol.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main()
