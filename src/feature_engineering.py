
import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD TRAINING AND TESTING DATA
# =====================================================

raw_train_df = pd.read_csv(TRAIN_PATH)
raw_test_df = pd.read_csv(TEST_PATH)

raw_train_df["date"] = pd.to_datetime(raw_train_df["date"])
raw_test_df["date"] = pd.to_datetime(raw_test_df["date"])

raw_train_df = raw_train_df.sort_values("date").reset_index(drop=True)
raw_test_df = raw_test_df.sort_values("date").reset_index(drop=True)

print("Training dataset shape:", raw_train_df.shape)
print("Testing dataset shape:", raw_test_df.shape)


# =====================================================
# 3. DATE AND TIME FEATURES
# =====================================================

def create_time_features(df):

    # Monday = 0, Sunday = 6
    df["Day_of_week"] = df["date"].dt.dayofweek

    # Extract hour and minute
    df["Hour"] = df["date"].dt.hour
    df["Minute"] = df["date"].dt.minute

    # Weekday = 0, Weekend = 1
    df["WeekStatus"] = (
        df["Day_of_week"] >= 5
    ).astype(int)

    # Convert time into decimal hours
    decimal_hour = df["Hour"] + df["Minute"] / 60

    # Cyclic time features
    df["Hour_sin"] = np.sin(
        2 * np.pi * decimal_hour / 24
    )

    df["Hour_cos"] = np.cos(
        2 * np.pi * decimal_hour / 24
    )

    return df


# =====================================================
# 4. LAGGED FEATURES
# =====================================================

lag_periods = {
    "10min": 1,
    "30min": 3,
    "1h": 6,
    "3h": 18,
    "24h": 144
}


def create_lag_features(df):

    for label, periods in lag_periods.items():

        df[f"Appliances_lag_{label}"] = (
            df["Appliances"].shift(periods)
        )

    return df


# =====================================================
# 5. ROLLING WINDOW FEATURES
# =====================================================

rolling_windows = {
    "1h": 6,
    "3h": 18
}


def create_rolling_features(df):

    # Exclude current target value
    previous_values = df["Appliances"].shift(1)

    for label, window in rolling_windows.items():

        df[f"rolling_mean_{label}"] = (
            previous_values.rolling(window=window).mean()
        )

        df[f"rolling_std_{label}"] = (
            previous_values.rolling(window=window).std()
        )

        df[f"rolling_min_{label}"] = (
            previous_values.rolling(window=window).min()
        )

        df[f"rolling_max_{label}"] = (
            previous_values.rolling(window=window).max()
        )

    return df


# =====================================================
# 6. GRADIENT AND ACCELERATION FEATURES
# =====================================================

def create_gradient_features(df):

    # Only use observations before the current target row
    previous_values = df["Appliances"].shift(1)

    # -------------------------------------------------
    # First-order gradients
    # -------------------------------------------------

    # Change over the previous 10 minutes
    df["gradient_10min"] = (
        previous_values.diff(1)
    )

    # Average change per 10-minute interval over 30 min
    df["gradient_30min"] = (
        previous_values - previous_values.shift(3)
    ) / 3

    # Average change per 10-minute interval over 1 hour
    df["gradient_1h"] = (
        previous_values - previous_values.shift(6)
    ) / 6

    # Average change per 10-minute interval over 3 hours
    df["gradient_3h"] = (
        previous_values - previous_values.shift(18)
    ) / 18

    # -------------------------------------------------
    # Second-order gradient (acceleration)
    # -------------------------------------------------

    df["acceleration_10min"] = (
        df["gradient_10min"].diff(1)
    )

    # -------------------------------------------------
    # Percentage change over 30 minutes
    # -------------------------------------------------

    df["percentage_change_30min"] = (
        previous_values.pct_change(periods=3) * 100
    )

    # -------------------------------------------------
    # Variability of recent changes
    # -------------------------------------------------

    df["gradient_std_30min"] = (
        df["gradient_10min"].rolling(window=3).std()
    )

    # -------------------------------------------------
    # Energy range over previous 30 minutes
    # -------------------------------------------------

    df["energy_range_30min"] = (
        previous_values.rolling(window=3).max()
        -
        previous_values.rolling(window=3).min()
    )

    return df


# =====================================================
# 7. APPLY FEATURE ENGINEERING
# =====================================================

def apply_feature_engineering(df):

    df = df.copy()

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_gradient_features(df)

    return df


# =====================================================
# 8. ENGINEER TRAINING DATA
# =====================================================

train_df = apply_feature_engineering(raw_train_df)

# Remove rows with insufficient historical observations
train_df = train_df.dropna().reset_index(drop=True)


# =====================================================
# 9. ENGINEER TESTING DATA
# =====================================================

# Use the original training data as historical context.
# This ensures the first test rows have valid lag,
# rolling and gradient features.

history_length = max(
    max(lag_periods.values()),
    max(rolling_windows.values()),
    18  # Maximum gradient window
)

history_df = raw_train_df.iloc[-history_length:].copy()

combined_test_history = pd.concat(
    [history_df, raw_test_df],
    ignore_index=True
)

# Generate features across the train/test boundary
combined_test_history = apply_feature_engineering(
    combined_test_history
)

# Keep only actual test observations
test_df = combined_test_history.iloc[
    history_length:
].copy()

test_df = test_df.reset_index(drop=True)


# =====================================================
# 10. VERIFY MISSING VALUES
# =====================================================

print("\n========== MISSING VALUE CHECK ==========")

print("\nTraining missing values:")
print(train_df.isnull().sum().sum())

print("\nTesting missing values:")
print(test_df.isnull().sum().sum())


# =====================================================
# 11. SAVE UPDATED DATASETS
# =====================================================

train_output_path = OUTPUT_DIR / "train_feature_added.csv"
test_output_path = OUTPUT_DIR / "test_feature_added.csv"

train_df.to_csv(
    train_output_path,
    index=False
)

test_df.to_csv(
    test_output_path,
    index=False
)


# =====================================================
# 12. DISPLAY ENGINEERED FEATURES
# =====================================================

new_features = [

    # Date and time
    "date",
    "Day_of_week",
    "WeekStatus",
    "Hour",
    "Hour_sin",
    "Hour_cos",

    # Lag features
    "Appliances_lag_10min",
    "Appliances_lag_30min",
    "Appliances_lag_1h",
    "Appliances_lag_3h",
    "Appliances_lag_24h",

    # Rolling features
    "rolling_mean_1h",
    "rolling_std_1h",
    "rolling_min_1h",
    "rolling_max_1h",
    "rolling_mean_3h",
    "rolling_std_3h",
    "rolling_min_3h",
    "rolling_max_3h",

    # Gradient features
    "gradient_10min",
    "gradient_30min",
    "gradient_1h",
    "gradient_3h",

    # Acceleration
    "acceleration_10min",

    # Additional temporal features
    "percentage_change_30min",
    "gradient_std_30min",
    "energy_range_30min"
]

print("\nTraining engineered features:")
print(train_df[new_features].head(10))

print("\nTesting engineered features:")
print(test_df[new_features].head(10))


# =====================================================
# 13. FINAL SUMMARY
# =====================================================

print("\n========================================")
print("FEATURE ENGINEERING COMPLETED")
print("========================================")

print("\nTraining dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

print("\nNew gradient features:")

gradient_features = [
    "gradient_10min",
    "gradient_30min",
    "gradient_1h",
    "gradient_3h",
    "acceleration_10min",
    "percentage_change_30min",
    "gradient_std_30min",
    "energy_range_30min"
]

for feature in gradient_features:
    print(f"- {feature}")

print("\nSaved files:")
print(train_output_path)
print(test_output_path)

print("\nCompleted successfully!")
