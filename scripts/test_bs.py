from src.analytics.black_scholes import bs_price_european

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

call = bs_price_european(S0, K, T, r, sigma, "call")
put = bs_price_european(S0, K, T, r, sigma, "put")

print("Call price:", round(call, 6))
print("Put price :", round(put, 6))

# Put-call parity check: C - P = S0 - K*e^{-rT}
lhs = call - put
rhs = S0 - K * (2.718281828459045 ** (-r * T))  # approx exp for parity check
print("Parity LHS:", round(lhs, 6))
print("Parity RHS:", round(rhs, 6))
