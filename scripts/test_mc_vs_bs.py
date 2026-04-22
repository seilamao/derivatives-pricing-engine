from src.analytics.black_scholes import bs_price_european
from src.engines.monte_carlo import mc_price_european

def main():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    option_type = "call"

    bs = bs_price_european(S0, K, T, r, sigma, option_type)
    mc, se = mc_price_european(S0, K, T, r, sigma, option_type, n_paths=200_000, seed=42)

    print(f"Black–Scholes {option_type}: {bs:.6f}")
    print(f"Monte Carlo  {option_type}: {mc:.6f}")
    print(f"Std error (MC): {se:.6f}")
    print(f"Abs error: {abs(mc - bs):.6f}")

    # Optional: show if BS is within a rough 95% CI of MC
    lo, hi = mc - 1.96 * se, mc + 1.96 * se
    print(f"Approx 95% CI: [{lo:.6f}, {hi:.6f}]")
    print("BS inside CI?", lo <= bs <= hi)

if __name__ == "__main__":
    main()
