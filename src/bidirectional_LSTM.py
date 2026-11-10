import os
import random
import json
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

SEED = 42

SEQUENCE_LENGTH = 24       # 24 observations = 4 hours
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_RATIO = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_feature_added.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_feature_added.csv"

OUTPUT_DIR = BASE_DIR / "models" / "hsbufc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. REPRODUCIBILITY
# ==========================================================

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

random.seed(SEED)
np.random.seed(SEED)

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ==========================================================
# 3. LOAD DATA
# ==========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

train_df = train_df.sort_values("date").reset_index(drop=True)
test_df = test_df.sort_values("date").reset_index(drop=True)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# ==========================================================
# 4. PREPARE FEATURES
# ==========================================================

TARGET = "Appliances"

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError(f"Target column '{TARGET}' is missing.")

# Use numeric columns only.
# Appliances is intentionally included as an input feature:
# historical consumption can help predict future consumption.

feature_columns = [
    col for col in train_df.columns
    if col != "date"
    and pd.api.types.is_numeric_dtype(train_df[col])
]

# Ensure both datasets have the same features in the same order.
missing_columns = [
    col for col in feature_columns
    if col not in test_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in test dataset: {missing_columns}"
    )

X_train = train_df[feature_columns].to_numpy(dtype=np.float32)
X_test = test_df[feature_columns].to_numpy(dtype=np.float32)

y_train = train_df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
y_test = test_df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)

if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
    raise ValueError("Input features contain NaN or infinite values.")

if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
    raise ValueError("Target contains NaN or infinite values.")

print("Number of input features:", len(feature_columns))
print("Features:", feature_columns)


# ==========================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ==========================================================

split_index = int(len(X_train) * (1 - VALIDATION_RATIO))

if split_index <= SEQUENCE_LENGTH:
    raise ValueError("Not enough training observations for this sequence length.")

print("\nTraining observations:", split_index)
print("Validation observations:", len(X_train) - split_index)


# ==========================================================
# 6. FEATURE AND TARGET SCALING
# ==========================================================

# Fit scalers ONLY using the training portion.
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_scaler.fit(X_train[:split_index])
target_scaler.fit(y_train[:split_index])

# Transform training and test features using the same scalers.
X_train_scaled = feature_scaler.transform(X_train).astype(np.float32)
X_test_scaled = feature_scaler.transform(X_test).astype(np.float32)

y_train_scaled = target_scaler.transform(y_train).astype(np.float32)


# ==========================================================
# 7. CREATE SEQUENCES
# ==========================================================

def create_training_sequences(X, y, sequence_length, split_index):

    X_train_sequences = []
    y_train_sequences = []

    X_val_sequences = []
    y_val_sequences = []

    for i in range(sequence_length, len(X)):

        sequence = X[i - sequence_length:i]
        target = y[i]

        if i < split_index:

            X_train_sequences.append(sequence)
            y_train_sequences.append(target)

        else:

            X_val_sequences.append(sequence)
            y_val_sequences.append(target)

    return (
        np.array(X_train_sequences, dtype=np.float32),
        np.array(y_train_sequences, dtype=np.float32),
        np.array(X_val_sequences, dtype=np.float32),
        np.array(y_val_sequences, dtype=np.float32)
    )


(
    X_train_seq,
    y_train_seq,
    X_val_seq,
    y_val_seq
) = create_training_sequences(
    X_train_scaled,
    y_train_scaled,
    SEQUENCE_LENGTH,
    split_index
)

print("\nSequence shapes:")

print("X_train:", X_train_seq.shape)
print("y_train:", y_train_seq.shape)

print("X_validation:", X_val_seq.shape)
print("y_validation:", y_val_seq.shape)


# ==========================================================
# 8. BUILD HSBUFC-STYLE MODEL
# ==========================================================

n_features = X_train_seq.shape[2]

model = Sequential([

    Input(shape=(SEQUENCE_LENGTH, n_features)),

    # Bidirectional LSTM
    Bidirectional(
        LSTM(
            64,
            activation="tanh",
            return_sequences=True
        )
    ),

    # First unidirectional LSTM
    LSTM(
        32,
        activation="tanh",
        return_sequences=True
    ),

    Dropout(0.2, seed=SEED + 1),

    # Second unidirectional LSTM
    LSTM(
        16,
        activation="tanh"
    ),

    Dropout(0.2, seed=SEED + 2),

    # Fully connected layers
    Dense(
        16,
        activation="relu"
    ),

    Dense(1)

])


# ==========================================================
# 9. COMPILE MODEL
# ==========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mae",
    metrics=["mae"]
)

model.summary()


# ==========================================================
# 10. CALLBACKS
# ==========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=str(OUTPUT_DIR / "best_model.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ==========================================================
# 11. TRAIN MODEL
# ==========================================================

print("\nStarting HSBUFC-style model training...")

history = model.fit(

    X_train_seq,
    y_train_seq,

    validation_data=(
        X_val_seq,
        y_val_seq
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    shuffle=False,
    verbose=1

)


# ==========================================================
# 12. SAVE TRAINING HISTORY
# ==========================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)


# ==========================================================
# 13. ROLLING ONE-STEP-AHEAD TESTING
# ==========================================================

print("\nPreparing test sequences...")

# Combine training and testing observations.
# This allows the first test prediction to use the
# final historical observations from the training set.

X_combined_scaled = np.concatenate(
    [X_train_scaled, X_test_scaled],
    axis=0
)

train_length = len(X_train_scaled)

X_test_sequences = []

for i in range(train_length, len(X_combined_scaled)):

    sequence = X_combined_scaled[
        i - SEQUENCE_LENGTH:i
    ]

    X_test_sequences.append(sequence)


X_test_sequences = np.array(
    X_test_sequences,
    dtype=np.float32
)

print("Test sequence shape:", X_test_sequences.shape)


# ==========================================================
# 14. PREDICTIONS
# ==========================================================

print("\nGenerating predictions...")

predictions_scaled = model.predict(
    X_test_sequences,
    verbose=1
)

# Convert predictions back to original Wh scale.
predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual_values = y_test.flatten()


# ==========================================================
# 15. EVALUATION
# ==========================================================

mae = mean_absolute_error(
    actual_values,
    predictions
)

mse = mean_squared_error(
    actual_values,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual_values,
    predictions
)


print("\n======================================")
print("HSBUFC-STYLE MODEL RESULTS")
print("======================================")

print(f"MAE  : {mae:.4f} Wh")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} Wh")
print(f"R2   : {r2:.4f}")


# ==========================================================
# 16. SAVE PREDICTIONS
# ==========================================================

results_df = pd.DataFrame({

    "date": test_df["date"],

    "Actual_Appliances": actual_values,

    "Predicted_Appliances": predictions,

    "Absolute_Error": np.abs(
        actual_values - predictions
    )

})

results_df.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ==========================================================
# 17. SAVE MODEL AND SCALERS
# ==========================================================

model.save(
    OUTPUT_DIR / "hsbufc_model.keras"
)

joblib.dump(
    feature_scaler,
    OUTPUT_DIR / "feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    OUTPUT_DIR / "target_scaler.pkl"
)

joblib.dump(
    feature_columns,
    OUTPUT_DIR / "feature_columns.pkl"
)


# ==========================================================
# 18. SAVE EVALUATION METRICS
# ==========================================================

metrics = {

    "model": "HSBUFC-style",

    "sequence_length": SEQUENCE_LENGTH,

    "forecast_horizon": "10 minutes",

    "number_of_features": len(feature_columns),

    "MAE": float(mae),

    "MSE": float(mse),

    "RMSE": float(rmse),

    "R2": float(r2)

}

with open(
    OUTPUT_DIR / "metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\n======================================")
print("TRAINING AND EVALUATION COMPLETED")
print("======================================")

print("Saved outputs to:", OUTPUT_DIR)