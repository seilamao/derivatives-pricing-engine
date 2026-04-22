import numpy as np
from src.engines.monte_carlo import european_payoff


def mc_price_antithetic(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_paths: int,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Monte Carlo pricing using antithetic variates.
    n_paths must be even.
    """

    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic variates.")

    rng = np.random.default_rng(seed)
    half = n_paths // 2

    Z = rng.standard_normal(size=half)
    Z_full = np.concatenate([Z, -Z])

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)

    payoffs = european_payoff(ST, K, option_type)
    disc_payoffs = np.exp(-r * T) * payoffs

    price = float(disc_payoffs.mean())
    stderr = float(disc_payoffs.std(ddof=1) / np.sqrt(n_paths))

    return price, stderr
