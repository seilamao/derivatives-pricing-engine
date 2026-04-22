from math import log, sqrt, exp, erf


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_delta(S0: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Black–Scholes delta for European call/put.
    """
    opt = option_type.lower()
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if T <= 0.0:
        # At expiry, delta is discontinuous; this is a simple convention.
        if opt == "call":
            return 1.0 if S0 > K else 0.0
        else:
            return -1.0 if S0 < K else 0.0

    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    if opt == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0
