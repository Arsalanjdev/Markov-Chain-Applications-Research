import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x, mean, std_dev):
    # Gaussian (Normal) Distribution
    return np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))

def metropolis_algorithm(initial_state, iterations, proposal_std_dev, target_mean, target_std_dev):
    samples = [initial_state]
    current_state = initial_state

    for _ in range(iterations):
        # Propose a new state
        proposed_state = current_state + np.random.normal(scale=proposal_std_dev)

        # Calculate acceptance probability
        acceptance_prob = min(1, target_distribution(proposed_state, target_mean, target_std_dev) /
                                  target_distribution(current_state, target_mean, target_std_dev))

        # Accept or reject the proposed state
        if np.random.uniform(0, 1) < acceptance_prob:
            current_state = proposed_state

        samples.append(current_state)

    return np.array(samples)

# Parameters
target_mean = 0
target_std_dev = 1
initial_state = 0
iterations = 50000
proposal_std_dev = 0.5

# Generate samples using Metropolis algorithm
samples = metropolis_algorithm(initial_state, iterations, proposal_std_dev, target_mean, target_std_dev)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the target distribution
x = np.linspace(-5, 5, 1000)
plt.plot(x, target_distribution(x, target_mean, target_std_dev), label='Target Distribution (Gaussian)', color='r')

# Plot the Metropolis samples
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Metropolis Samples')

plt.title('Metropolis Algorithm: Approximating Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
