
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

input_path = Path("data/raw/energy_data_set.csv")
output_dir = Path("data/processed")

output_dir.mkdir(parents=True, exist_ok=True)

Z_SCORE_THRESHOLD = 3
TRAIN_RATIO = 0.80

# Load original dataset
df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. EXTRACT DAY INFORMATION
# ============================================================

df["Day_of_week"] = df["date"].dt.dayofweek
df["Day_name"] = df["date"].dt.day_name()

# One-hot encode day names
day_encoding = pd.get_dummies(
    df["Day_name"],
    prefix="Day",
    dtype=int
)

df = pd.concat(
    [df, day_encoding],
    axis=1
)


# ============================================================
# 3. EXTRACT WEEKEND INFORMATION
# ============================================================

df["Is_weekend"] = (
    df["Day_of_week"] >= 5
).astype(int)


# ============================================================
# 4. CONVERT TIME INTO SINE AND COSINE
# ============================================================

hour = df["date"].dt.hour
minute = df["date"].dt.minute

decimal_hour = hour + minute / 60

df["Hour_sin"] = np.sin(
    2 * np.pi * decimal_hour / 24
)

df["Hour_cos"] = np.cos(
    2 * np.pi * decimal_hour / 24
)


# ============================================================
# 5. REMOVE UNNECESSARY COLUMNS
# ============================================================

columns_to_remove = [
    "Hour",
    "Minute",
    "Day_of_week",
    "WeekStatus",
    "Day_name",
    
]

df.drop(
    columns=[
        col for col in columns_to_remove
        if col in df.columns
    ],
    inplace=True
)

# Remove target lag and rolling features if present
engineered_target_features = [
    col for col in df.columns
    if col.startswith("Appliances_lag_")
    or col.startswith("rolling_")
]

df.drop(
    columns=engineered_target_features,
    inplace=True
)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

target_column = "Appliances"

if target_column not in df.columns:
    raise ValueError("Appliances target column not found!")

dates = df["date"].copy()

X = df.drop(
    columns=["date", target_column]
)

y = df[target_column].copy()

# Keep numerical features only
X = X.select_dtypes(include=["number"])


# ============================================================
# 7. MISSING VALUE CHECK
# ============================================================

print("\n========== MISSING VALUE CHECK ==========")

print("\nMissing values in features:")

feature_missing = X.isnull().sum()

print(feature_missing[feature_missing > 0])

print("\nMissing values in target:")
print(y.isnull().sum())

total_missing = (
    X.isnull().sum().sum()
    + y.isnull().sum()
)

print("\nTotal missing values:", total_missing)

if total_missing > 0:
    raise ValueError(
        "Missing values detected. "
        "Handle them before proceeding."
    )

print("No missing values detected!")


# ============================================================
# 8. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(X) * TRAIN_RATIO)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

dates_train = dates.iloc[:split_index].copy()
dates_test = dates.iloc[split_index:].copy()

print("\n========== DATASET SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining date range:")
print(
    dates_train.iloc[0],
    "to",
    dates_train.iloc[-1]
)

print("\nTesting date range:")
print(
    dates_test.iloc[0],
    "to",
    dates_test.iloc[-1]
)


# ============================================================
# 9. Z-SCORE OUTLIER DETECTION
# ============================================================

# Combine features and target for outlier detection
zscore_data = pd.concat(
    [X, y.rename(target_column)],
    axis=1
)

# Exclude binary features from Z-score detection
continuous_columns = [
    col for col in zscore_data.columns
    if zscore_data[col].nunique() > 2
]

# Calculate mean and standard deviation
# using training data only
train_zscore_data = zscore_data.iloc[:split_index]

z_means = train_zscore_data[continuous_columns].mean()

z_stds = train_zscore_data[continuous_columns].std(
    ddof=0
)

z_stds = z_stds.replace(0, np.nan)

# Calculate Z-scores for all records
# using training statistics
z_scores = (
    zscore_data[continuous_columns] - z_means
) / z_stds

# Identify outliers
outlier_mask = (
    z_scores.abs() > Z_SCORE_THRESHOLD
)


# ============================================================
# 10. OUTLIER SUMMARY
# ============================================================

outlier_summary = pd.DataFrame({
    "Feature": continuous_columns,

    "Training_Outliers": [
        int(
            outlier_mask.iloc[:split_index][col].sum()
        )
        for col in continuous_columns
    ],

    "Testing_Outliers": [
        int(
            outlier_mask.iloc[split_index:][col].sum()
        )
        for col in continuous_columns
    ]
})

outlier_summary["Total_Outliers"] = (
    outlier_summary["Training_Outliers"]
    + outlier_summary["Testing_Outliers"]
)

outlier_summary = outlier_summary.sort_values(
    "Total_Outliers",
    ascending=False
)

print("\n========== Z-SCORE OUTLIER SUMMARY ==========")

print("Z-score threshold:", Z_SCORE_THRESHOLD)

print(
    outlier_summary.to_string(index=False)
)

outlier_summary.to_csv(
    output_dir / "zscore_outlier_summary.csv",
    index=False
)


# ============================================================
# 11. SAVE INDIVIDUAL OUTLIER RECORDS
# ============================================================

outlier_records = []

for col in continuous_columns:

    flagged_indices = np.where(
        outlier_mask[col].fillna(False)
    )[0]

    for idx in flagged_indices:

        outlier_records.append({
            "date": dates.iloc[idx],

            "Dataset_Split": (
                "train" if idx < split_index else "test"
            ),

            "Feature": col,

            "Original_Value": zscore_data[col].iloc[idx],

            "Z_Score": z_scores[col].iloc[idx]
        })


outlier_records_df = pd.DataFrame(
    outlier_records,
    columns=[
        "date",
        "Dataset_Split",
        "Feature",
        "Original_Value",
        "Z_Score"
    ]
)

outlier_records_df.to_csv(
    output_dir / "zscore_outlier_records.csv",
    index=False
)

print(
    "\nTotal individual outlier records:",
    len(outlier_records_df)
)

print("Outliers identified but NOT removed.")


# ============================================================
# 12. CREATE TRAINING DATASET
#     WITHOUT STANDARDIZATION
# ============================================================

train_df = X_train.copy()

train_df.insert(
    0,
    "date",
    dates_train.to_numpy()
)

# Keep original Appliances values in Wh
train_df["Appliances"] = y_train.to_numpy()

train_output_path = (
    output_dir / "train.csv"
)

train_df.to_csv(
    train_output_path,
    index=False
)


# ============================================================
# 13. CREATE TESTING DATASET
#     WITHOUT STANDARDIZATION
# ============================================================

test_df = X_test.copy()

test_df.insert(
    0,
    "date",
    dates_test.to_numpy()
)

# Keep original Appliances values in Wh
test_df["Appliances"] = y_test.to_numpy()

test_output_path = (
    output_dir / "test.csv"
)

test_df.to_csv(
    test_output_path,
    index=False
)


# ============================================================
# 14. VERIFY GENERATED DATASETS
# ============================================================

print("\n========== DATA VERIFICATION ==========")

print("\nTraining dataset shape:")
print(train_df.shape)

print("\nTesting dataset shape:")
print(test_df.shape)

print("\nTraining feature statistics:")
print(
    train_df[X.columns].describe().round(3)
)

print("\nTesting feature statistics:")
print(
    test_df[X.columns].describe().round(3)
)

print("\nTraining target statistics:")
print(
    train_df["Appliances"].describe()
)

print("\nTesting target statistics:")
print(
    test_df["Appliances"].describe()
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n============================================")
print("DEEP LEARNING DATASETS CREATED SUCCESSFULLY")
print("============================================")

print("\nTraining dataset:")
print(train_output_path)

print("Shape:", train_df.shape)

print("\nTesting dataset:")
print(test_output_path)

print("Shape:", test_df.shape)

print("\nSaved outlier reports:")

print(
    output_dir / "zscore_outlier_summary.csv"
)

print(
    output_dir / "zscore_outlier_records.csv"
)



print("\nCompleted successfully!")
