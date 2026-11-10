
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

OUTPUT_DIR = BASE_DIR / "model" / "randomforest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load training and testing datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

# Sort chronologically
train_df = train_df.sort_values("date").reset_index(drop=True)
test_df = test_df.sort_values("date").reset_index(drop=True)

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

# --------------------------------------------------
# 3. Separate input features and target
# --------------------------------------------------

X_train = train_df.drop(columns=["date", "Appliances"])
X_test = test_df.drop(columns=["date", "Appliances"])

y_train = train_df["Appliances"]
y_test = test_df["Appliances"]

# Keep numerical input features
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

# Ensure testing features match training features
X_test = X_test.reindex(columns=X_train.columns)

print("\nNumber of input features:", X_train.shape[1])

# --------------------------------------------------
# 4. Handle missing values
# --------------------------------------------------

# Remove rows with missing values from training data
valid_train = X_train.notna().all(axis=1) & y_train.notna()

X_train = X_train.loc[valid_train].reset_index(drop=True)
y_train = y_train.loc[valid_train].reset_index(drop=True)

# Remove rows with missing values from testing data
valid_test = X_test.notna().all(axis=1) & y_test.notna()

X_test = X_test.loc[valid_test].reset_index(drop=True)
y_test = y_test.loc[valid_test].reset_index(drop=True)

test_dates = test_df.loc[valid_test, "date"].reset_index(drop=True)

print("\nTraining samples after cleaning:", len(X_train))
print("Testing samples after cleaning:", len(X_test))

# --------------------------------------------------
# 5. Display date ranges
# --------------------------------------------------

print("\nTraining date range:")
print(
    train_df.loc[valid_train, "date"].iloc[0],
    "to",
    train_df.loc[valid_train, "date"].iloc[-1]
)

print("\nTesting date range:")
print(test_dates.iloc[0], "to", test_dates.iloc[-1])

# --------------------------------------------------
# 6. Train initial Random Forest
# --------------------------------------------------

initial_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features=1.0
)

initial_model.fit(X_train, y_train)

print("\nInitial Random Forest trained successfully!")

# --------------------------------------------------
# 7. Calculate feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": initial_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

# --------------------------------------------------
# 8. Plot feature importance
# --------------------------------------------------

plt.figure(figsize=(14, 12))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 9. Select top 15 features
# --------------------------------------------------

TOP_N = 15

selected_features = (
    importance_df.head(TOP_N)["Feature"].tolist()
)

print(f"\nSelected top {TOP_N} features:")

for feature in selected_features:
    print(feature)

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Save selected features
pd.DataFrame({
    "Selected_Features": selected_features
}).to_csv(
    OUTPUT_DIR / "selected_features.csv",
    index=False
)

# --------------------------------------------------
# 10. Train final Random Forest model
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features=1.0
)

rf_model.fit(X_train_selected, y_train)

print("\nFinal Random Forest model trained successfully!")

# --------------------------------------------------
# 11. Make predictions
# --------------------------------------------------

y_pred = rf_model.predict(X_test_selected)

# --------------------------------------------------
# 12. Evaluate model performance
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== RANDOM FOREST PERFORMANCE ==========")

print(f"MAE  : {mae:.4f} Wh")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} Wh")
print(f"R2   : {r2:.4f}")
print(f"R2   : {r2 * 100:.2f}%")

# --------------------------------------------------
# 13. Create prediction results
# --------------------------------------------------

results = pd.DataFrame({
    "date": test_dates.values,
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Error": y_test.values - y_pred
})

print("\nFirst 10 predictions:")
print(results.head(10))

# --------------------------------------------------
# 14. Actual vs Predicted time-series plot
# --------------------------------------------------

PLOT_SAMPLES = min(300, len(y_test))

plt.figure(figsize=(16, 7))

plt.plot(
    test_dates.iloc[:PLOT_SAMPLES],
    y_test.values[:PLOT_SAMPLES],
    label="Actual Consumption",
    linewidth=1.8
)

plt.plot(
    test_dates.iloc[:PLOT_SAMPLES],
    y_pred[:PLOT_SAMPLES],
    label="Predicted Consumption",
    linewidth=1.8
)

plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Appliances Consumption (Wh)")
plt.legend()
plt.grid(alpha=0.3)
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
# 15. Actual vs Predicted scatter plot
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

plt.xlabel("Actual Consumption (Wh)")
plt.ylabel("Predicted Consumption (Wh)")
plt.title("Random Forest: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 16. Residual distribution plot
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

plt.title("Random Forest: Residual Distribution")
plt.xlabel("Residual Error (Wh)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# --------------------------------------------------
# 17. Save feature importance
# --------------------------------------------------

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

# --------------------------------------------------
# 18. Save predictions
# --------------------------------------------------

results.to_csv(
    OUTPUT_DIR / "random_forest_predictions.csv",
    index=False
)

# --------------------------------------------------
# 19. Save evaluation metrics
# --------------------------------------------------

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2", "R2_Percentage"],
    "Value": [mae, mse, rmse, r2, r2 * 100]
})

metrics.to_csv(
    OUTPUT_DIR / "random_forest_metrics.csv",
    index=False
)

# --------------------------------------------------
# 20. Display final results
# --------------------------------------------------

print("\n========== EXECUTION COMPLETED ==========")

print("\nNumber of selected features:", len(selected_features))

print("\nSelected features:")
print(selected_features)

print("\nOutput directory:", OUTPUT_DIR)

print("\nGenerated files:")

for file in OUTPUT_DIR.iterdir():
    print(file.name)
