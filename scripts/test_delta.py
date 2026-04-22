from src.analytics.greeks import bs_delta
from src.greeks.delta_fd import delta_central_fd_common_random_numbers


def main():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    option_type = "call"

    # Finite difference step size (good default)
    h = 0.1  # you can also try 0.01*S0

    # Simulations
    n_paths = 200_000
    seed = 123

    delta_bs = bs_delta(S0, K, T, r, sigma, option_type)
    delta_mc = delta_central_fd_common_random_numbers(
        S0, K, T, r, sigma, option_type,
        n_paths=n_paths, h=h, seed=seed
    )

    print("Black–Scholes Delta:", round(delta_bs, 6))
    print("Monte Carlo FD Delta:", round(delta_mc, 6))
    print("Abs difference:", round(abs(delta_mc - delta_bs), 6))


if __name__ == "__main__":
    main()
