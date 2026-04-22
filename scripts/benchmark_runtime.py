import time
import numpy as np


def mc_loop(S0, K, T, r, sigma, n_paths):
    """
    Naive Python loop Monte Carlo (slow).
    """
    rng = np.random.default_rng(42)
    disc = np.exp(-r * T)

    total = 0.0
    for _ in range(n_paths):
        Z = rng.standard_normal()
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        payoff = max(ST - K, 0.0)
        total += payoff

    return disc * (total / n_paths)


def mc_vectorised(S0, K, T, r, sigma, n_paths):
    """
    Vectorised NumPy Monte Carlo (fast).
    """
    rng = np.random.default_rng(42)
    Z = rng.standard_normal(size=n_paths)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)

    return np.exp(-r * T) * payoffs.mean()


def main():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    n_loop = 200_000      # keep smaller so loop finishes
    n_vec = 2_000_000     # larger to show performance

    print("Running naive loop...")
    t0 = time.perf_counter()
    price_loop = mc_loop(S0, K, T, r, sigma, n_loop)
    t1 = time.perf_counter()

    print("Running vectorised...")
    t2 = time.perf_counter()
    price_vec = mc_vectorised(S0, K, T, r, sigma, n_vec)
    t3 = time.perf_counter()

    loop_time = t1 - t0
    vec_time = t3 - t2

    print("\nNaive Loop:")
    print("Price:", round(price_loop, 6))
    print("Time (seconds):", round(loop_time, 4))

    print("\nVectorised NumPy:")
    print("Price:", round(price_vec, 6))
    print("Time (seconds):", round(vec_time, 4))

    print("\nSpeedup factor:", round(loop_time / vec_time, 2), "x")


if __name__ == "__main__":
    main()
