
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# 1. Load training dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_feature_added.csv"
)

df = pd.read_csv(TRAIN_PATH)

# --------------------------------------------------
# 2. Create output directory
# --------------------------------------------------

OUTPUT_DIR = BASE_DIR / "output" / "linearregression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3. Identify numerical features
# --------------------------------------------------

target = "Appliances"

numerical_features = [
    col for col in df.select_dtypes(include="number").columns
]

# --------------------------------------------------
# 4. Calculate Pearson correlation matrix
# --------------------------------------------------

correlation_matrix = df[numerical_features].corr(
    method="pearson"
)

# --------------------------------------------------
# 5. Calculate feature correlation with Appliances
# --------------------------------------------------

correlations = correlation_matrix[target].drop(target)

correlation_table = pd.DataFrame({
    "Feature": correlations.index,
    "Correlation": correlations.values,
    "Absolute_Correlation": correlations.abs().values
})

correlation_table = correlation_table.sort_values(
    by="Absolute_Correlation",
    ascending=False
).reset_index(drop=True)

# --------------------------------------------------
# 6. Save correlation table
# --------------------------------------------------

correlation_table.to_csv(
    OUTPUT_DIR / "linear_regression_correlation.csv",
    index=False
)

correlation_matrix.to_csv(
    OUTPUT_DIR / "linear_regression_correlation_matrix.csv"
)

# --------------------------------------------------
# 7. Feature correlation bar chart
# --------------------------------------------------

plt.figure(figsize=(14, 12))

colors = [
    "steelblue" if value >= 0 else "indianred"
    for value in correlation_table["Correlation"]
]

plt.barh(
    correlation_table["Feature"],
    correlation_table["Correlation"],
    color=colors
)

plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel("Pearson Correlation", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title(
    "Feature Correlation with Appliances",
    fontsize=16,
    pad=15
)

plt.xticks(fontsize=10)
plt.yticks(fontsize=9)
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 8. Complete correlation matrix heatmap
# --------------------------------------------------

plt.figure(figsize=(22, 18))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.4,
    annot_kws={"size": 7},
    cbar_kws={"label": "Pearson Correlation"}
)

plt.title(
    "Pearson Correlation Matrix",
    fontsize=18,
    pad=20
)

plt.xticks(rotation=90, fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\nCorrelation analysis completed successfully!")

print("\nFeature correlation with Appliances:")
print(
    correlation_table[
        ["Feature", "Correlation"]
    ].to_string(index=False)
)

print("\nOutput directory:", OUTPUT_DIR)

print("\nGenerated files:")
for file in OUTPUT_DIR.iterdir():
    print(file.name)
