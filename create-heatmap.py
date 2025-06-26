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

# Step 3: Define coverage values (raw values)
raw_data = {
    "Reference":            [1.0, 1.0, 1.0, 0.66],
    "Non-deterministic":    [1.0, 1.0, 1.0, 0.33],
    "Rounds incorrect":     [0.0, 1.0, 1.0, 0.0],
    "Client disconnect":    [1.0, 1.0, 1.0, 1.0]
}

# Step 4: Format data for plotting
df = pd.DataFrame(raw_data, index=transitions)
df_percent = df.applymap(lambda x: f"{int(x * 100)}%")

# Step 5: Define nicer 4-color map (can adjust to taste)
colors = ["#e74c3c", "#f1c40f", "#e67e22", "#2ecc71"]  # red, yellow, orange, green
boundaries = [0.0, 0.01, 0.34, 0.67, 1.01]
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=len(colors))

# Step 6: Plot heatmap
plt.figure(figsize=(8, 5))
ax = sns.heatmap(df, annot=df_percent, fmt="", cmap=cmap, norm=norm,
                 cbar=False, linewidths=1, linecolor="black")

plt.title("Transition Coverage Heatmap per Test Case", fontsize=14)
plt.xlabel("Test Case", fontsize=12)
plt.ylabel("Transition", fontsize=12)

# Optional: Add custom legend
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=colors[0], label="0%"),
    mpatches.Patch(color=colors[1], label="33%"),
    mpatches.Patch(color=colors[2], label="66%"),
    mpatches.Patch(color=colors[3], label="100%")
]
# plt.legend(handles=legend_patches, title="Coverage", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
