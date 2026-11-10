
# ============================================================
# 0. DETERMINISTIC CONFIGURATION
# ============================================================

import os

SEED = 42

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# 2. SET RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

tf.keras.utils.set_random_seed(SEED)

tf.config.experimental.enable_op_determinism()

print("Random seed:", SEED)
print("Deterministic operations enabled.")


# ============================================================
# 3. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = (
    BASE_DIR / "data" / "processed" /
    "train_selected_features.csv"
)

TEST_CSV = (
    BASE_DIR / "data" / "processed" /
    "test_selected_features.csv"
)

OUTPUT_DIR = BASE_DIR / "models" / "cnn_lstm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CNN-LSTM configuration
SEQUENCE_LENGTH = 3
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

if list(train_df.columns) != list(test_df.columns):
    raise ValueError("Training and testing columns do not match.")

if train_df.isnull().sum().sum() > 0:
    raise ValueError("Missing values found in training data.")

if test_df.isnull().sum().sum() > 0:
    raise ValueError("Missing values found in testing data.")


# ============================================================
# 6. SELECT INPUT FEATURES
# ============================================================

# Exclude date because it is metadata.
# Keep Appliances as an input feature.

excluded_columns = ["date"]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

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

y_train_raw = train_df["Appliances"].to_numpy(
    dtype=np.float32
)

y_test_raw = test_df["Appliances"].to_numpy(
    dtype=np.float32
)

print("\nNumber of input features:", len(feature_columns))
print("Input features:", feature_columns)


# ============================================================
# 7. NO STANDARDIZATION
# ============================================================

# Use the values directly from the CSV files.
# No StandardScaler or other scaling is applied.

X_train = X_train_raw
X_test = X_test_raw

y_train = y_train_raw
y_test = y_test_raw

print("\nUsing existing dataset values without standardization.")


# ============================================================
# 8. CREATE TRAINING SEQUENCES
# ============================================================

X_train_sequences = []
y_train_sequences = []

for i in range(SEQUENCE_LENGTH, len(X_train)):

    X_train_sequences.append(
        X_train[i - SEQUENCE_LENGTH:i]
    )

    y_train_sequences.append(
        y_train[i]
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
# 9. CREATE TESTING SEQUENCES
# ============================================================

# Use the final training observations as historical context.

history_data = X_train[-SEQUENCE_LENGTH:]

combined_test_data = np.concatenate(
    [history_data, X_test],
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

y_test_sequences = y_test.copy()

print("\nTesting sequence shape:", X_test_sequences.shape)
print("Testing target shape:", y_test_sequences.shape)

if len(X_test_sequences) != len(y_test_sequences):
    raise ValueError(
        "Testing sequence and target counts do not match."
    )


# ============================================================
# 10. BUILD CNN-LSTM MODEL
# ============================================================

tf.keras.backend.clear_session()

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            X_train_sequences.shape[2]
        )
    ),

    # CNN feature extraction
    Conv1D(
        filters=64,
        kernel_size=2,
        padding="same",
        activation="relu"
    ),

    Conv1D(
        filters=32,
        kernel_size=2,
        padding="same",
        activation="relu"
    ),

    # Temporal learning
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

    # Regression output
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
# 11. CALLBACKS
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
# 12. TRAIN MODEL
# ============================================================

history = model.fit(

    X_train_sequences,
    y_train_sequences,

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
# 13. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test_sequences,
    verbose=0
).flatten()


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test_sequences,
    predictions
)

mse = mean_squared_error(
    y_test_sequences,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test_sequences,
    predictions
)

print("\n===================================")
print("         CNN-LSTM RESULTS")
print("===================================")

print(f"MAE  : {mae:.4f} Wh")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} Wh")
print(f"R2   : {r2:.4f}")

print("===================================")


# ============================================================
# 15. SAVE EVALUATION METRICS
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
    OUTPUT_DIR / "cnn_lstm_metrics.csv",
    index=False
)


# ============================================================
# 16. PLOT TRAINING AND VALIDATION LOSS
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

plt.title("CNN-LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (Wh squared)")
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
# 17. PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    y_test_sequences[:500],
    label="Actual Consumption",
    linewidth=1.5
)

plt.plot(
    predictions[:500],
    label="Predicted Consumption",
    linewidth=1.5
)

plt.title(
    "CNN-LSTM: Actual vs Predicted Appliances Consumption"
)

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
# 18. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "date": test_df["date"],

    "Actual_Wh": y_test_sequences,

    "Predicted_Wh": predictions

})

results.to_csv(
    OUTPUT_DIR / "cnn_lstm_predictions.csv",
    index=False
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

model.save(
    OUTPUT_DIR / "cnn_lstm_energy_model.keras"
)


# ============================================================
# 20. COMPLETION
# ============================================================

print("\nCNN-LSTM training completed successfully!")

print("\nGenerated model outputs:")
print("- cnn_lstm_energy_model.keras")
print("- cnn_lstm_predictions.csv")
print("- cnn_lstm_metrics.csv")
print("- training_loss.png")
print("- actual_vs_predicted.png")

print("\nOutput directory:", OUTPUT_DIR)
