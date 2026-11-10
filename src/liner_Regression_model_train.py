
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_feature_added.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_feature_added.csv"
)

OUTPUT_DIR = BASE_DIR / "output" / "linearregression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load training and testing datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Convert date columns
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

# Sort chronologically
train_df = train_df.sort_values("date").reset_index(drop=True)
test_df = test_df.sort_values("date").reset_index(drop=True)

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

# --------------------------------------------------
# 3. Features to exclude
# --------------------------------------------------

features_to_remove = [
    "rolling_max_3h",
    "rolling_std_3h",
    "rolling_mean_3h",
    "Appliances_lag_30min",
    "Appliances_lag_24h",
    "Appliances_lag_3h",
    "Appliances_lag_1h",
    "rolling_min_3h",
    "rv1",
    "rv2",
    "Minute",
    "Visibility",
    "RH_4",
    "RH_5",
    "WeekStatus",
    "Is_weekend"
]

# Check whether requested features exist
for name, df in [
    ("Training", train_df),
    ("Testing", test_df)
]:
    missing_features = [
        col for col in features_to_remove
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name} dataset is missing features: "
            f"{missing_features}"
        )

# Remove unwanted features
train_df = train_df.drop(columns=features_to_remove)
test_df = test_df.drop(columns=features_to_remove)

print("\nRemoved features:", len(features_to_remove))

# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X_train = train_df.drop(columns=["date", "Appliances"])
y_train = train_df["Appliances"]

X_test = test_df.drop(columns=["date", "Appliances"])
y_test = test_df["Appliances"]

# Ensure training and testing columns match
X_test = X_test[X_train.columns]

print("\nNumber of input features:", X_train.shape[1])
print("Target variable: Appliances")

# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError("Missing values found in input features.")

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError("Missing values found in target variable.")

print("\nMissing value check passed.")

# --------------------------------------------------
# 6. Display date ranges
# --------------------------------------------------

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining date range:")
print(train_df["date"].iloc[0], "to", train_df["date"].iloc[-1])

print("\nTesting date range:")
print(test_df["date"].iloc[0], "to", test_df["date"].iloc[-1])

# --------------------------------------------------
# 7. Standardize numerical input features
# --------------------------------------------------

scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data using the training scaler
X_test_scaled = scaler.transform(X_test)

print("\nFeature standardization completed!")

# Save fitted scaler
joblib.dump(
    scaler,
    OUTPUT_DIR / "linear_regression_scaler.pkl"
)

# --------------------------------------------------
# 8. Train Linear Regression model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train_scaled, y_train)

print("\nLinear Regression model trained successfully!")

# --------------------------------------------------
# 9. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)

# --------------------------------------------------
# 10. Evaluate model performance
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.4f} Wh")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} Wh")
print(f"R2   : {r2:.4f}")
print(f"R2   : {r2 * 100:.2f}%")

# --------------------------------------------------
# 11. Create prediction results
# --------------------------------------------------

results = pd.DataFrame({
    "date": test_df["date"].values,
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Error": y_test.values - y_pred
})

print("\nFirst 10 predictions:")
print(results.head(10))

# --------------------------------------------------
# 12. Actual vs Predicted time-series plot
# --------------------------------------------------

plt.figure(figsize=(16, 7))

plt.plot(
    test_df["date"].iloc[:300],
    y_test.values[:300],
    label="Actual Appliances Consumption",
    linewidth=1.8
)

plt.plot(
    test_df["date"].iloc[:300],
    y_pred[:300],
    label="Predicted Appliances Consumption",
    linewidth=1.8
)

plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Appliances Energy Consumption (Wh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 13. Actual vs Predicted scatter plot
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Consumption (Wh)")
plt.ylabel("Predicted Consumption (Wh)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 14. Residual distribution plot
# --------------------------------------------------

residuals = y_test.values - y_pred

plt.figure(figsize=(10, 6))

plt.hist(
    residuals,
    bins=40,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2
)

plt.title("Distribution of Prediction Errors (Residuals)")
plt.xlabel("Residual Error (Wh)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 15. Save predictions
# --------------------------------------------------

PREDICTION_PATH = (
    OUTPUT_DIR / "linear_regression_predictions.csv"
)

results.to_csv(PREDICTION_PATH, index=False)

# --------------------------------------------------
# 16. Save model performance metrics
# --------------------------------------------------

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2", "R2_Percentage"],
    "Value": [mae, mse, rmse, r2, r2 * 100]
})

metrics.to_csv(
    OUTPUT_DIR / "linear_regression_metrics.csv",
    index=False
)

# --------------------------------------------------
# 17. Display final results
# --------------------------------------------------

print("\n========== EXECUTION COMPLETED ==========")

print("\nInput features used:")
print(X_train.columns.tolist())

print("\nOutput directory:", OUTPUT_DIR)

print("\nGenerated files:")

for file in OUTPUT_DIR.iterdir():
    print(file.name)
