from src.analytics.black_scholes import bs_price_european
from src.engines.monte_carlo import mc_price_european
from src.engines.variance_reduction import mc_price_antithetic


def main():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    option_type = "call"
    n_paths = 100_000
    seed = 42

    bs = bs_price_european(S0, K, T, r, sigma, option_type)

    mc_price, mc_se = mc_price_european(
        S0, K, T, r, sigma, option_type, n_paths=n_paths, seed=seed
    )

    anti_price, anti_se = mc_price_antithetic(
        S0, K, T, r, sigma, option_type, n_paths=n_paths, seed=seed
    )

    print("Black–Scholes:", round(bs, 6))
    print("\nStandard Monte Carlo")
    print("Price:", round(mc_price, 6))
    print("Std Error:", round(mc_se, 6))

    print("\nAntithetic Monte Carlo")
    print("Price:", round(anti_price, 6))
    print("Std Error:", round(anti_se, 6))

    print("\nStd Error Reduction:",
          round((mc_se - anti_se) / mc_se * 100, 2), "%")


if __name__ == "__main__":
    main()
