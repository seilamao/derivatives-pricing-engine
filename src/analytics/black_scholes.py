import numpy as np
from math import erf, sqrt, exp, log


def norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function (no SciPy dependency)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price_european(S0: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Black–Scholes price for a European call or put.

    Parameters
    ----------
    S0 : float
        Spot price at time 0
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Continuously-compounded risk-free rate
    sigma : float
        Volatility (annualised)
    option_type : str
        'call' or 'put'
    """
    opt = option_type.lower()
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    # Handle edge cases cleanly
    if T <= 0.0:
        return max(S0 - K, 0.0) if opt == "call" else max(K - S0, 0.0)

    if sigma <= 0.0:
        # Deterministic under risk-neutral measure if sigma=0
        ST = S0 * exp(r * T)
        payoff = max(ST - K, 0.0) if opt == "call" else max(K - ST, 0.0)
        return exp(-r * T) * payoff

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if opt == "call":
        return S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    else:  # put
        return K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
