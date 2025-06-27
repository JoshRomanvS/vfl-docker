import matplotlib.pyplot as plt
import numpy as np
from runs import deterministic_run, non_deterministic_runs

rounds = list(range(1, 51))

# Convert non-deterministic runs to a NumPy array for easy manipulation
non_det_array = np.array([run[:50] for run in non_deterministic_runs])

# Compute mean and spread (min/max or standard deviation)
mean_non_det = np.mean(non_det_array, axis=0)
min_non_det = np.min(non_det_array, axis=0)
max_non_det = np.max(non_det_array, axis=0)

# Plotting
plt.figure(figsize=(10, 6))

# Plot deterministic run
plt.plot(rounds, deterministic_run[:50], label='Deterministic Run', color='blue')

# Plot mean of non-deterministic runs
plt.plot(rounds, mean_non_det, label='Mean of Non-deterministic Runs', color='orange')

# Plot shaded area (spread) — between min and max
plt.fill_between(rounds, min_non_det, max_non_det, color='gray', alpha=0.3, label='Spread of Non-deterministic Runs')

# Labels and formatting
plt.title("Accuracy over Rounds: Deterministic vs Non-deterministic")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()
