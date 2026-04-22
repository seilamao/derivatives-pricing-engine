import numpy as np
from src.models.gbm import simulate_gbm_terminal


def european_payoff(ST: np.ndarray, K: float, option_type: str) -> np.ndarray:
    opt = option_type.lower()
    if opt == "call":
        return np.maximum(ST - K, 0.0)
    if opt == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def mc_price_european(
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
    Monte Carlo price for a European option under risk-neutral GBM.

    Returns
    -------
    price : float
        Discounted Monte Carlo estimator
    stderr : float
        Standard error of the estimator (discounted)
    """
    ST = simulate_gbm_terminal(S0=S0, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed)
    payoffs = european_payoff(ST, K, option_type)
    disc_payoffs = np.exp(-r * T) * payoffs

    price = float(np.mean(disc_payoffs))
    stderr = float(np.std(disc_payoffs, ddof=1) / np.sqrt(n_paths))
    return price, stderr
