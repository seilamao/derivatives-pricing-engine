import numpy as np
import matplotlib.pyplot as plt

from src.analytics.black_scholes import bs_price_european
from src.models.gbm import simulate_gbm_paths, simulate_gbm_terminal
from src.engines.monte_carlo import european_payoff


def mc_price_from_terminal(ST: np.ndarray, K: float, r: float, T: float, option_type: str) -> float:
    payoff = european_payoff(ST, K, option_type)
    return float(np.exp(-r * T) * payoff.mean())


def main():
    # Parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    option_type = "call"
    seed = 42

    bs = bs_price_european(S0, K, T, r, sigma, option_type)
    print(f"Black–Scholes {option_type}: {bs:.6f}")

    # Ensure figures folder exists (safe even if it already exists)
    import os
    os.makedirs("figures", exist_ok=True)

    # Plot 1: sample paths
    n_steps = 252
    n_paths_plot = 20
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps=n_steps, n_paths=n_paths_plot, seed=seed)
    t = np.linspace(0, T, n_steps + 1)

    plt.figure()
    for i in range(n_paths_plot):
        plt.plot(t, paths[i])
    plt.title("GBM sample paths (risk-neutral)")
    plt.xlabel("t (years)")
    plt.ylabel("S(t)")
    plt.tight_layout()
    plt.savefig("figures/paths.png", dpi=200)
    plt.close()

    # Convergence experiment
    sims_grid = np.array([500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000])

    mc_prices = []
    abs_errors = []

    for i, M in enumerate(sims_grid):
        ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths=int(M), seed=seed + i)
        mc = mc_price_from_terminal(ST, K, r, T, option_type)
        mc_prices.append(mc)
        abs_errors.append(abs(mc - bs))

    mc_prices = np.array(mc_prices)
    abs_errors = np.array(abs_errors)

    # Plot 2: convergence
    plt.figure()
    plt.plot(sims_grid, mc_prices, marker="o")
    plt.axhline(bs, linestyle="--")
    plt.xscale("log")
    plt.title("Monte Carlo price estimate vs number of simulations")
    plt.xlabel("Number of simulations (log scale)")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("figures/convergence.png", dpi=200)
    plt.close()

    # Plot 3: error vs sims (log-log)
    plt.figure()
    plt.plot(sims_grid, abs_errors, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Absolute pricing error vs number of simulations")
    plt.xlabel("Number of simulations (log scale)")
    plt.ylabel("|MC - BS| (log scale)")
    plt.tight_layout()
    plt.savefig("figures/error_vs_sims.png", dpi=200)
    plt.close()

    # Print table (for report/README)
    print("\nM      MC Price     |MC-BS|")
    for M, p, e in zip(sims_grid, mc_prices, abs_errors):
        print(f"{M:>6d}  {p:>10.6f}  {e:>8.6f}")

    print("\nSaved figures to:")
    print(" - figures/paths.png")
    print(" - figures/convergence.png")
    print(" - figures/error_vs_sims.png")


if __name__ == "__main__":
    main()
