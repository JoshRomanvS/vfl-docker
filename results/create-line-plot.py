import matplotlib.pyplot as plt
from runs import non_deterministic_runs

rounds = list(range(1, 51))

# Sort the 10 runs by their first value (starting accuracy)
sorted_runs = sorted(non_deterministic_runs, key=lambda run: run[0])

# Pick 5 evenly spread-out ones (e.g., lowest, low-mid, mid, high-mid, highest)
selected_runs = [sorted_runs[i] for i in [0, 1, 3, 4, 8, 9]]  # or adjust indices as needed

# Plotting
plt.figure(figsize=(10, 6))

for i, run in enumerate(selected_runs):
    plt.plot(rounds, run[:50], label=f'Non-deterministic Run {i+1}')

# Labels and formatting
plt.title("Non-deterministic Runs with Spread-Out Starting Accuracies")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.grid(True)
# plt.legend()
plt.tight_layout()

plt.show()
