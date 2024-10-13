import numpy as np
import matplotlib.pyplot as plt

# Generate a large sample from the Exp(1) distribution
X = np.random.exponential(1, 100)


# Define the simple function approximation f_n(x)
def f_n(x, n):
    # Initialize total sum to zero
    total_sums = np.zeros_like(x)

    # Loop through k intervals and approximate f(x) = sqrt(x) using step functions
    for k in range(0, n * 2 ** n + 1):
        lower_bound = k * 2 ** -n
        upper_bound = (k + 1) * 2 ** -n
        total_sums += np.where((x >= lower_bound) & (x < upper_bound), np.sqrt(lower_bound), 0)

    # For values greater than n, set f_n(x) to sqrt(n)
    total_sums += np.where(x >= n, np.sqrt(n), 0)

    return total_sums


# Computing the expected values E[f_n] for n = 1 to 15
expected_values = []
n_values = range(1, 16)

for n in n_values:
    f_n_values = f_n(X, n)
    expected_values.append(np.mean(f_n_values))

# Part (c): Plot the sequence of expected values and the theoretical value
plt.figure(figsize=(8, 6))

# Plot the stepwise expected values E[f_n]
plt.step(n_values, expected_values, where='mid', label='E[f_n]', marker='o', color='blue')

# Plot the theoretical expected value E[f(x)] = sqrt(pi)/2
theoretical_value = np.sqrt(np.pi) / 2
plt.axhline(y=theoretical_value, color='orange', linestyle='-', label='E[f(X)] = sqrt(pi)/2')

# Labels and title
plt.xlabel('Index n of Simple Function Sequence')
plt.ylabel('Expected Value E[f_n]')
plt.title('Approximation of Expected Value by Simple Functions')

# Add a legend
plt.legend()

# Save the plot for expected values
plt.savefig("expected_values_plot.png")
plt.show()

# Part (c): Visualizing the approximation by simple functions for n = 1 to 10
x_values = np.linspace(0, 10, 500)  # Generate x values for visualization

# Plotting the true function f(x) = sqrt(x)
plt.figure(figsize=(10, 6))
# Loop to generate and save plots for n = 1 to 10
for n in range(1, 11):
    plt.step(x_values, np.sqrt(x_values), label='f(x) = sqrt(x)', color='orange', lw=2)
    plt.step(x_values, f_n(x_values, n), where='mid', label=f'f_{n}(x)', lw=1)

    # Labels and title for each plot
    plt.xlabel('x')
    plt.ylabel('f_n(x)')
    plt.title(f'Approximation of sqrt(x) by Simple Function f_{n}(x)')

    # Legend for the individual plots
    plt.legend(loc='best')
    plt.show()