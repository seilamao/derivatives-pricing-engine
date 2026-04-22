import numpy as np
import matplotlib.pyplot as plt

print("Project setup works.")
print("NumPy version:", np.__version__)

x = np.linspace(0, 1, 5)
print("NumPy test array:", x)

# tiny plot test (should pop up a window)
plt.plot(x, x**2)
plt.title("Matplotlib sanity check")
plt.show()
