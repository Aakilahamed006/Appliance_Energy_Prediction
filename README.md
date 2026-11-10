# Appliance Energy Prediction

An end-to-end Machine Learning and Deep Learning pipeline for forecasting household appliance electrical energy consumption (Wh) at 10-minute intervals using indoor environmental sensors, local meteorological conditions, and temporal time-series dynamics.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Project Directory Structure](#project-directory-structure)
4. [Environment Setup](#environment-setup)
5. [End-to-End Execution Guide](#end-to-end-execution-guide)
   - [Step 1: Exploratory Data Analysis](#step-1-exploratory-data-analysis)
   - [Step 2: Data Preprocessing & Outlier Detection](#step-2-data-preprocessing--outlier-detection)
   - [Step 3: Feature Engineering](#step-3-feature-engineering)
   - [Step 4: Feature Selection (Linear Models)](#step-4-feature-selection-linear-models)
   - [Step 5: Machine Learning Models](#step-5-machine-learning-models)
   - [Step 6: Deep Learning Time-Series Models](#step-6-deep-learning-time-series-models)
6. [Model Benchmark & Performance Comparison](#model-benchmark--performance-comparison)
7. [Saved Artifacts & Outputs](#saved-artifacts--outputs)
8. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Overview

Accurate prediction of appliance energy consumption enables smart home energy optimization, peak demand shaving, and dynamic load scheduling. This project explores the **UCI Appliance Energy Prediction Dataset**, featuring:
- **Chronological time-series data** recorded every 10 minutes over ~4.5 months.
- **Indoor temperature & humidity monitoring** across 9 distinct rooms.
- **On-site & nearby weather station metrics** (temperature, humidity, atmospheric pressure, wind speed, visibility, and dew point).
- **Comparative modeling** spanning baseline linear regression, tree ensembles (Random Forest), and deep temporal architectures (LSTM, CNN-LSTM, and Bidirectional LSTM).

---

## Dataset Description

- **Source**: UCI Machine Learning Repository (`data/raw/energy_data_set.csv`)
- **Total Records**: 19,735 observations
- **Target Variable**: `Appliances` (Energy use in Wh)
- **Features**:
  - `date`: Time stamp (every 10 minutes)
  - `lights`: Energy consumption of lighting fixtures in Wh
  - `T1`–`T9`: Temperature in degrees Celsius (°C) across 9 indoor zones (Kitchen, Living room, Laundry, Office, Bathroom, North building outdoor, Ironing room, Teenager room, Parents room)
  - `RH_1`–`RH_9`: Relative humidity percentages (%) in the same 9 zones
  - `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`: Meteorological conditions recorded at Chievres airport weather station
  - `rv1`, `rv2`: Non-dimensional random benchmark variables (removed during feature selection)

---

## Project Directory Structure

```text
Appliance_Energy_Prediction/
│
├── data/
│   ├── raw/
│   │   └── energy_data_set.csv                      # Original raw dataset
│   └── processed/
│       ├── train.csv                                # Chronological train split
│       ├── test.csv                                 # Chronological test split
│       ├── train_feature_added.csv                  # Train set with lags & rolling stats
│       ├── test_feature_added.csv                   # Test set with lags & rolling stats
│       ├── train_selected_features.csv              # Filtered training features
│       ├── test_selected_features.csv               # Filtered testing features
│       ├── deep_learning_train.pkl                  # Serialized DL train sequences
│       ├── deep_learning_test.pkl                   # Serialized DL test sequences
│       ├── deep_learning_feature_scaler.pkl         # Scaler for inputs
│       └── deep_learning_target_scaler.pkl          # Scaler for target variable
│
├── notebooks/
│   ├── EDA.ipynb                                    # Exploratory Data Analysis notebook
│   └── appliance_power_consumption.png              # Historical energy trend visualization
│
├── src/
│   ├── __init__.py                                  # Package initialization
│   ├── data_preprocessing.py                        # Outlier analysis, temporal splits, scaling
│   ├── feature_engineering.py                       # Lag generation, rolling windows, cyclical time
│   ├── feature_selection_linear_regression.py       # Correlation filtering & multicollinearity removal
│   ├── liner_Regression_model_train.py              # Linear Regression training & residual analysis
│   ├── Random_forset.py                             # Random Forest regressor & feature importance
│   ├── LSTM_model_train.py                          # 2-layer stacked LSTM network
│   ├── cnn_LSTM.py                                  # CNN-LSTM hybrid model
│   └── bidirectional_LSTM.py                        # Bidirectional LSTM (HSBUFC architecture)
│
├── models/
│   ├── lstm/                                        # LSTM trained weights, metrics, and plots
│   ├── cnn_lstm/                                    # CNN-LSTM trained weights, metrics, and plots
│   └── hsbufc/                                      # Bidirectional LSTM weights, metrics, and scalers
│
├── output/
│   ├── linearregression/                            # Correlation matrices, error plots, and predictions
│   └── randomforest/                                # Feature importance charts, residuals, predictions
│
├── requirements.txt                                 # Core dependencies
└── README.md                                        # Documentation and execution guide
```

---

## Environment Setup

### 1. Clone or Open the Repository
Ensure you are in the project root directory:
```bash
cd "c:\Users\CCS\Desktop\victorian labs\Appliance_Energy_Prediction"
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# (Or Command Prompt):
# .venv\Scripts\activate.bat

# (Or Linux / macOS):
# source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register Jupyter Kernel
To execute notebooks inside VS Code / Jupyter Lab:
```bash
python -m ipykernel install --user --name appliance-energy --display-name "Python (Appliance Energy)"
```

---

## End-to-End Execution Guide

Execute the pipeline sequentially using your activated virtual environment or the full path to `.venv/Scripts/python.exe`.

### Step 1: Exploratory Data Analysis
Open `notebooks/EDA.ipynb` in your IDE or JupyterLab:
- Select the registered kernel (`Python (Appliance Energy)` or `.venv`).
- Visualizes consumption distributions, daily/weekly seasonality, room temperature/humidity distributions, and correlation maps.

---

### Step 2: Data Preprocessing & Outlier Detection
Parses timestamps, performs Z-score outlier analysis ($Z > 3$), standardizes features, and creates an 80/20 chronological train-test split:
```bash
python src/data_preprocessing.py
```
**Outputs**:
- `data/processed/train.csv` & `data/processed/test.csv`
- Outlier summary and record logs in `data/processed/`

---

### Step 3: Feature Engineering
Computes time-series lags, rolling window statistics, and cyclical trigonometric features (to prevent temporal boundary discontinuities):
- **Lags**: 10 min, 30 min, 1 hour, 3 hours, 24 hours.
- **Rolling Stats**: 1-hour and 3-hour rolling mean, standard deviation, min, and max.
- **Cyclic Encoding**: Sine and cosine representations of hour, minute, and day of week.
```bash
python src/feature_engineering.py
```
**Outputs**:
- `data/processed/train_feature_added.csv`
- `data/processed/test_feature_added.csv`

---

### Step 4: Feature Selection (Linear Models)
Removes redundant variables, random noise columns (`rv1`, `rv2`), and highly collinear features:
```bash
python src/feature_selection_linear_regression.py
```
**Outputs**:
- `data/processed/train_selected_features.csv`
- `data/processed/test_selected_features.csv`

---

### Step 5: Machine Learning Models

#### 5.1. Linear Regression
Trains an Ordinary Least Squares (OLS) model with standardized features, computes error metrics, plots actual vs. predicted curves, and evaluates residuals:
```bash
python src/liner_Regression_model_train.py
```
**Outputs stored in `output/linearregression/`**:
- `linear_regression_metrics.csv`
- `linear_regression_predictions.csv`
- `actual_vs_predicted.png`
- `correlation_matrix.png`
- `residual_distribution.png`

#### 5.2. Random Forest Regressor
Fits an ensemble of 200 decision trees, extracts Gini feature importances, selects top predictive features, and trains the final model:
```bash
python src/Random_forset.py
```
**Outputs stored in `output/randomforest/`**:
- `feature_importance.png` & `feature_importance.csv`
- `random_forest_metrics.csv`
- `random_forest_predictions.csv`
- `actual_vs_predicted.png`

---

### Step 6: Deep Learning Time-Series Models

#### 6.1. Stacked LSTM (2-Layer LSTM)
Trains a 2-layer LSTM (64 and 32 hidden units) with Dropout (0.2) using sequence lengths of 144 steps (24 hours lookback):
```bash
python src/LSTM_model_train.py
```
**Outputs stored in `models/lstm/`**:
- Model: `lstm_energy_model.keras`
- Metrics: `lstm_metrics.csv`
- Plots: `training_loss.png`, `actual_vs_predicted.png`

#### 6.2. CNN-LSTM Hybrid Model
Applies 1D Convolutional layers for local temporal pattern extraction followed by LSTM layers for long-range temporal sequence modeling:
```bash
python src/cnn_LSTM.py
```
**Outputs stored in `models/cnn_lstm/`**:
- Model: `cnn_lstm_energy_model.keras`
- Metrics: `cnn_lstm_metrics.csv`
- Plots: `training_loss.png`, `actual_vs_predicted.png`

#### 6.3. Bidirectional LSTM (HSBUFC Architecture)
Implements forward and backward information flow to capture historical context and seasonal periodicity:
```bash
python src/bidirectional_LSTM.py
```
**Outputs stored in `models/hsbufc/`**:
- Model: `best_model.keras` & `hsbufc_model.keras`
- Scalers: `feature_scaler.pkl`, `target_scaler.pkl`
- Metrics: `metrics.json`
- Plots & Histories: `training_history.csv`, `predictions.csv`

---

## Model Benchmark & Performance Comparison

All models were evaluated on the held-out test split (chronologically partitioned to prevent future data leakage):

| Model Architecture | MAE (Wh) | RMSE (Wh) | MSE | $R^2$ Score | Key Strengths / Characteristics |
|---|---|---|---|---|---|
| **CNN-LSTM Hybrid** | **26.24** | **58.64** | **3438.64** | **58.51%** | **Best overall accuracy**; Combines local 1D feature extraction with temporal memory |
| **Linear Regression** (Engineered Features) | **27.22** | **60.00** | **3599.48** | **56.57%** | Fast, interpretable; Performs strongly with engineered lag and rolling variables |
| **Bidirectional LSTM** (HSBUFC) | **28.57** | **66.21** | **4384.00** | **47.10%** | Captures bidirectional context with fast sequence convergence |
| **Stacked LSTM** (2-Layer) | **28.74** | **63.94** | **4088.59** | **50.67%** | Strong sequence stability over 24-hour historical windows |
| **Random Forest Regressor** (Top 15 Features) | **42.6029** | **77.6114** | **6023.53** | **27.32%** | Highlights key room sensors (`T3`, `RH_1`, `T8`), but suffers on non-stationary shifts |

> **Key Observation**: Hybrid deep learning (**CNN-LSTM**) and well-engineered linear regression with historical lag/rolling statistics achieve the highest predictive accuracy ($\sim 58.5\%$ variance explained) on 10-minute resolution energy forecasts.

---

## Saved Artifacts & Outputs

- **Trained Deep Learning Models**:
  - `models/cnn_lstm/cnn_lstm_energy_model.keras`
  - `models/lstm/lstm_energy_model.keras`
  - `models/hsbufc/best_model.keras`
- **Model Scalers**:
  - `models/hsbufc/feature_scaler.pkl` & `target_scaler.pkl`
  - `output/linearregression/linear_regression_scaler.pkl`
- **Prediction Files**:
  - `models/cnn_lstm/cnn_lstm_predictions.csv`
  - `models/lstm/lstm_predictions.csv`
  - `output/linearregression/linear_regression_predictions.csv`
  - `output/randomforest/random_forest_predictions.csv`
- **Evaluation Visualizations**:
  - Training loss and convergence curves (`training_loss.png`)
  - Actual vs. predicted time-series overlays (`actual_vs_predicted.png`)
  - Feature importance bar charts (`feature_importance.png`)
  - Error residual distributions (`residual_distribution.png`)

---

## Troubleshooting & FAQs

### 1. Headless / Server Execution (Non-GUI Terminals)
If running on a remote server or terminal without a display, Matplotlib may show warnings on `plt.show()`. Set the non-interactive backend:
```powershell
$env:MPLBACKEND="Agg"
python src/LSTM_model_train.py
```

### 2. Relative Path Errors (`FileNotFoundError`)
Always run commands from the project root (`Appliance_Energy_Prediction`). All scripts use `pathlib.Path(__file__).resolve().parent.parent` to locate data directories reliably.

### 3. GPU vs CPU Training
TensorFlow 2.15+ on native Windows runs on CPU by default (which takes $\approx 1$–$2$ minutes for sequence training). If NVIDIA GPU acceleration is needed, run inside **WSL2** (Windows Subsystem for Linux) with CUDA/cuDNN enabled.

