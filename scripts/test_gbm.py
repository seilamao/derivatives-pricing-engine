import numpy as np
import matplotlib.pyplot as plt

from src.models.gbm import simulate_gbm_paths, simulate_gbm_terminal

S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0

# 1) Quick shape check
paths = simulate_gbm_paths(S0, r, sigma, T, n_steps=252, n_paths=5, seed=123)
print("paths shape:", paths.shape)  # expected (5, 253)

# 2) Risk-neutral expectation sanity check: E[S_T] = S0 * exp(rT)
n_paths = 200_000
ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths=n_paths, seed=1)
mc_mean = ST.mean()
theory_mean = S0 * np.exp(r * T)
print("E[ST] MC   :", mc_mean)
print("E[ST] theory:", theory_mean)
print("relative error:", abs(mc_mean - theory_mean) / theory_mean)

# 3) Plot a few paths (should look reasonable)
paths_plot = simulate_gbm_paths(S0, r, sigma, T, n_steps=252, n_paths=20, seed=42)
t = np.linspace(0, T, 253)

plt.figure()
for i in range(paths_plot.shape[0]):
    plt.plot(t, paths_plot[i])
plt.title("GBM sample paths (risk-neutral)")
plt.xlabel("t (years)")
plt.ylabel("S_t")
plt.tight_layout()
plt.show()
