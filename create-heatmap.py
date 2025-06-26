import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Define the transitions
transitions = [
    "idle → idle",
    "idle → training",
    "training → training",
    "training → idle"
]

# Step 2: Define the test cases
test_cases = [
    "Reference",
    "Non-deterministic",
    "Rounds incorrect",
    "Client disconnect"
]

# Step 3: Fill in transition coverage
# Use: 0 = red (not used), 0.5 = orange (partially used), 1 = green (fully used)
# Placeholder values – replace after analyzing your images
data = {
    "Reference":            [1, 1, 1, 1],
    "Non-deterministic":    [1, 1, 1, 1],
    "Rounds incorrect":     [0.5, 1, 1, 0],
    "Client disconnect":    [0, 1, 0.5, 1],
}

# Step 4: Create DataFrame
df = pd.DataFrame(data, index=transitions)

# Step 5: Plot heatmap
plt.figure(figsize=(8, 5))
ax = sns.heatmap(df, annot=True, cmap=sns.color_palette(["red", "orange", "green"], as_cmap=True),
                 cbar=False, linewidths=1, linecolor="black", fmt=".1f")

plt.title("Transition Coverage Heatmap per Test Case", fontsize=14)
plt.xlabel("Test Case", fontsize=12)
plt.ylabel("Transition", fontsize=12)
plt.tight_layout()
plt.show()
