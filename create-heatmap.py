import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

# Step 1: Define transitions
transitions = [
    "idle → idle",
    "idle → training",
    "training → training",
    "training → idle"
]

# Step 2: Define test cases
test_cases = [
    "Reference",
    "Non-deterministic",
    "Rounds incorrect",
    "Client disconnect"
]

# Step 3: Define coverage values (placeholder until all images are processed)
# 0.0 = red, 0.33 = yellow, 0.66 = orange, 1.0 = green
data = {
    "Reference":            [1.0, 1.0, 1.0, 0.0],
    "Non-deterministic":    [1.0, 1.0, 1.0, 1.0],  # placeholder
    "Rounds incorrect":     [1.0, 1.0, 1.0, 0.33], # placeholder
    "Client disconnect":    [1.0, 1.0, 0.33, 0.66] # placeholder
}

# Step 4: Create DataFrame
df = pd.DataFrame(data, index=transitions)

# Step 5: Define 4-color map
colors = ["red", "yellow", "orange", "green"]
boundaries = [0.0, 0.01, 0.34, 0.67, 1.01]  # bins for 0, 1/3, 2/3, 3/3
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=len(colors))

# Step 6: Plot heatmap
plt.figure(figsize=(8, 5))
ax = sns.heatmap(df, annot=True, cmap=cmap, norm=norm,
                 cbar=False, linewidths=1, linecolor="black", fmt=".2f")

plt.title("Transition Coverage Heatmap per Test Case", fontsize=14)
plt.xlabel("Test Case", fontsize=12)
plt.ylabel("Transition", fontsize=12)
plt.tight_layout()
plt.show()
