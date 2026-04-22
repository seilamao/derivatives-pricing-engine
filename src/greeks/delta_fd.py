import numpy as np
from src.engines.monte_carlo import european_payoff


def mc_price_terminal_given_Z(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    Z: np.ndarray,
) -> float:
    """
    MC price using *fixed* standard normal draws Z (common random numbers).
    """
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = european_payoff(ST, K, option_type)
    return float(np.exp(-r * T) * payoffs.mean())


def delta_central_fd_common_random_numbers(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_paths: int,
    h: float,
    seed: int | None = None,
) -> float:
    """
    Central finite-difference delta:
        (V(S0+h) - V(S0-h)) / (2h)

    Uses common random numbers (same Z) to reduce variance.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=n_paths)

    up = mc_price_terminal_given_Z(S0 + h, K, T, r, sigma, option_type, Z)
    down = mc_price_terminal_given_Z(S0 - h, K, T, r, sigma, option_type, Z)

    return (up - down) / (2.0 * h)
