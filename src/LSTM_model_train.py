
# ============================================================
# 0. DETERMINISTIC CONFIGURATION
# ============================================================

import os

SEED = 42

# Set environment variables BEFORE importing TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 2. SET RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

# Set Python, NumPy and TensorFlow seeds together
tf.keras.utils.set_random_seed(SEED)

# Enable deterministic TensorFlow operations
tf.config.experimental.enable_op_determinism()

print("Random seed:", SEED)
print("Deterministic TensorFlow operations enabled.")


# ============================================================
# 3. PATHS AND CONFIGURATION
# ============================================================

# Project root directory (one level above src)
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = BASE_DIR / "data" / "processed" / "train_feature_added.csv"
TEST_CSV = BASE_DIR / "data" / "processed" / "test_feature_added.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "lstm"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Processed datasets
TRAIN_PKL = PROCESSED_DIR / "deep_learning_train.pkl"
TEST_PKL = PROCESSED_DIR / "deep_learning_test.pkl"

# Scalers
FEATURE_SCALER_PATH = (
    PROCESSED_DIR / "deep_learning_feature_scaler.pkl"
)

TARGET_SCALER_PATH = (
    PROCESSED_DIR / "deep_learning_target_scaler.pkl"
)

# LSTM configuration
SEQUENCE_LENGTH = 3  # Previous 30 minutes (3 x 10 minutes)
EPOCHS = 50
BATCH_SIZE = 64


# ============================================================
# 4. LOAD TRAINING AND TESTING DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

train_df = train_df.sort_values("date").reset_index(drop=True)
test_df = test_df.sort_values("date").reset_index(drop=True)

print("\nTraining dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nTraining period:")
print(train_df["date"].iloc[0], "to", train_df["date"].iloc[-1])

print("\nTesting period:")
print(test_df["date"].iloc[0], "to", test_df["date"].iloc[-1])


# ============================================================
# 5. VALIDATE DATA
# ============================================================

if "Appliances" not in train_df.columns:
    raise ValueError("Appliances column missing from training data.")

if "Appliances" not in test_df.columns:
    raise ValueError("Appliances column missing from testing data.")

if len(train_df) <= SEQUENCE_LENGTH:
    raise ValueError("Insufficient training observations.")

if len(test_df) == 0:
    raise ValueError("Testing dataset is empty.")

# Ensure both datasets have the same columns
if list(train_df.columns) != list(test_df.columns):
    raise ValueError(
        "Training and testing columns do not match."
    )

# Ensure there are no missing values
if train_df.isnull().sum().sum() > 0:
    raise ValueError("Missing values found in training data.")

if test_df.isnull().sum().sum() > 0:
    raise ValueError("Missing values found in testing data.")


# ============================================================
# 6. SELECT INPUT FEATURES
# ============================================================

# Exclude date because it is metadata.
# Keep Appliances as an input feature.

excluded_columns = [
    "date"
]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

# Check that all selected features are numeric
non_numeric_columns = [
    col for col in feature_columns
    if not pd.api.types.is_numeric_dtype(train_df[col])
]

if non_numeric_columns:
    raise ValueError(
        f"Non-numeric columns found: {non_numeric_columns}"
    )

X_train_raw = train_df[feature_columns].to_numpy(
    dtype=np.float32
)

X_test_raw = test_df[feature_columns].to_numpy(
    dtype=np.float32
)

# Target
y_train_raw = train_df[["Appliances"]].to_numpy(
    dtype=np.float32
)

y_test_raw = test_df[["Appliances"]].to_numpy(
    dtype=np.float32
)

print("\nNumber of input features:", len(feature_columns))
print("Input features:", feature_columns)


# ============================================================
# 7. STANDARDIZE INPUT FEATURES
# ============================================================

# Fit scaler ONLY on training data
feature_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(
    X_train_raw
).astype(np.float32)

# Transform test data using training scaler
X_test_scaled = feature_scaler.transform(
    X_test_raw
).astype(np.float32)

# Save feature scaler
joblib.dump(
    feature_scaler,
    FEATURE_SCALER_PATH
)

print("\nInput feature standardization completed.")


# ============================================================
# 8. STANDARDIZE TARGET
# ============================================================

target_scaler = StandardScaler()

# Fit only on training target
y_train_scaled = target_scaler.fit_transform(
    y_train_raw
).astype(np.float32)

# Transform test target using training target scaler
y_test_scaled = target_scaler.transform(
    y_test_raw
).astype(np.float32)

# Save target scaler
joblib.dump(
    target_scaler,
    TARGET_SCALER_PATH
)

print("Target standardization completed.")


# ============================================================
# 9. CREATE TRAINING SEQUENCES
# ============================================================

X_train_sequences = []
y_train_sequences = []

for i in range(SEQUENCE_LENGTH, len(train_df)):

    # Previous time steps
    X_train_sequences.append(
        X_train_scaled[i - SEQUENCE_LENGTH:i]
    )

    # Target at the next time step
    y_train_sequences.append(
        y_train_scaled[i]
    )

X_train_sequences = np.array(
    X_train_sequences,
    dtype=np.float32
)

y_train_sequences = np.array(
    y_train_sequences,
    dtype=np.float32
)

print("\nTraining sequence shape:", X_train_sequences.shape)
print("Training target shape:", y_train_sequences.shape)


# ============================================================
# 10. CREATE TESTING SEQUENCES
# ============================================================

# Use the final training observations as historical context
history_data = X_train_scaled[-SEQUENCE_LENGTH:]

# Combine training history with standardized test features
combined_test_data = np.concatenate(
    [history_data, X_test_scaled],
    axis=0
)

X_test_sequences = []

for i in range(SEQUENCE_LENGTH, len(combined_test_data)):

    X_test_sequences.append(
        combined_test_data[i - SEQUENCE_LENGTH:i]
    )

X_test_sequences = np.array(
    X_test_sequences,
    dtype=np.float32
)

y_test_sequences = y_test_scaled.copy()

print("\nTesting sequence shape:", X_test_sequences.shape)
print("Testing target shape:", y_test_sequences.shape)

if len(X_test_sequences) != len(y_test_sequences):
    raise ValueError(
        "Testing sequence and target counts do not match."
    )


# ============================================================
# 11. SAVE SEPARATE TRAINING AND TESTING DATASETS
# ============================================================

train_dataset = {

    "X": X_train_sequences,

    "y": y_train_sequences,

    "dates": train_df["date"].iloc[
        SEQUENCE_LENGTH:
    ].to_numpy(),

    "feature_columns": feature_columns,

    "sequence_length": SEQUENCE_LENGTH
}

joblib.dump(
    train_dataset,
    TRAIN_PKL
)

test_dataset = {

    "X": X_test_sequences,

    "y": y_test_sequences,

    "dates": test_df["date"].to_numpy(),

    "feature_columns": feature_columns,

    "sequence_length": SEQUENCE_LENGTH
}

joblib.dump(
    test_dataset,
    TEST_PKL
)

print("\nProcessed datasets saved successfully.")


# ============================================================
# 12. LOAD PROCESSED DATASETS
# ============================================================

train_dataset = joblib.load(TRAIN_PKL)
test_dataset = joblib.load(TEST_PKL)

X_train = train_dataset["X"]
y_train = train_dataset["y"]

X_test = test_dataset["X"]
y_test = test_dataset["y"]

print("\nLoaded processed datasets.")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 13. CLEAR PREVIOUS MODEL STATE
# ============================================================

tf.keras.backend.clear_session()

# Reset seeds before model initialization
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 14. BUILD LSTM MODEL
# ============================================================

model = Sequential([

    Input(
        shape=(SEQUENCE_LENGTH, X_train.shape[2])
    ),

    LSTM(
        64,
        activation="tanh",
        return_sequences=True
    ),

    Dropout(
        0.2,
        seed=SEED + 1
    ),

    LSTM(
        32,
        activation="tanh"
    ),

    Dropout(
        0.2,
        seed=SEED + 2
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ============================================================
# 15. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.0001
)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

history = model.fit(

    X_train,
    y_train,

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    validation_split=0.2,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    shuffle=False,
    verbose=1
)


# ============================================================
# 17. MAKE PREDICTIONS
# ============================================================

predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 18. CONVERT PREDICTIONS BACK TO WH
# ============================================================

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)

predictions_wh = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual_wh = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()


# ============================================================
# 19. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    actual_wh,
    predictions_wh
)

mse = mean_squared_error(
    actual_wh,
    predictions_wh
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual_wh,
    predictions_wh
)

print("\n===================================")
print("           LSTM RESULTS")
print("===================================")

print(f"MAE  : {mae:.4f} Wh")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} Wh")
print(f"R2   : {r2:.4f}")

print("===================================")


# ============================================================
# 20. SAVE EVALUATION METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],

    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})

metrics_df.to_csv(
    OUTPUT_DIR / "lstm_metrics.csv",
    index=False
)


# ============================================================
# 21. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Standardized MSE Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 22. PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    actual_wh[:500],
    label="Actual Consumption",
    linewidth=1.5
)

plt.plot(
    predictions_wh[:500],
    label="Predicted Consumption",
    linewidth=1.5
)

plt.title("LSTM: Actual vs Predicted Appliances Consumption")
plt.xlabel("Time Steps (10-minute intervals)")
plt.ylabel("Energy Consumption (Wh)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 23. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "date": test_dataset["dates"],

    "Actual_Wh": actual_wh,

    "Predicted_Wh": predictions_wh

})

results.to_csv(
    OUTPUT_DIR / "lstm_predictions.csv",
    index=False
)


# ============================================================
# 24. SAVE MODEL
# ============================================================

model.save(
    OUTPUT_DIR / "lstm_energy_model.keras"
)


# ============================================================
# 25. COMPLETION
# ============================================================

print("\nLSTM training completed successfully!")

print("\nGenerated processed datasets:")
print("- deep_learning_train.pkl")
print("- deep_learning_test.pkl")

print("\nGenerated scaler files:")
print("- deep_learning_feature_scaler.pkl")
print("- deep_learning_target_scaler.pkl")

print("\nGenerated model outputs:")
print("- lstm_energy_model.keras")
print("- lstm_predictions.csv")
print("- lstm_metrics.csv")
print("- training_loss.png")
print("- actual_vs_predicted.png")

print("\nOutput directory:", OUTPUT_DIR)
