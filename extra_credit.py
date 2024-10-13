import numpy as np
import matplotlib.pyplot as plt

# Generate samples of X ~ Exp(1) random variables
X = np.random.exponential(1, 100000)  # Using more samples for a better approximation

# Define the simple function f_n(x)
def f_n(x, n):
    totals = np.zeros_like(x)
    for k in range(0, (n * 2**n)+1):
        lower_bound = k * 2**-n
        upper_bound = (k + 1) * 2**-n
        totals += np.where((x >= lower_bound) & (x < upper_bound), np.sqrt(lower_bound), 0)
    totals += np.where(x >= n, np.sqrt(n), 0)
    return totals

# Compute expected values for n = 1 to 15
expected_values = []
n_values = range(1, 16)
for n in n_values:
    f_n_values = f_n(X, n)
    expected_values.append(np.mean(f_n_values))

# Plotting the sequence of expected values
plt.step(n_values, expected_values, where='mid', label='E[f_n]', marker='o', color='blue')
plt.axhline(y=np.sqrt(np.pi)/2, color='r', linestyle='-', label='E[f] = sqrt(pi)/2')
plt.xlabel('Index n of Simple Function Sequence')
plt.ylabel('Expected Value E[f_n]')
plt.title('Approximation of Expected Value by Simple Functions')
plt.legend()
plt.show()
