"""
utils.py
Shared preprocessing helpers for the Mobile Price Prediction ANN project.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATASET_DIR = "dataset"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "preprocessing", "scaler.pkl")


def load_and_prepare_data():
    """
    Loads train.csv, splits into train/validation sets, and scales features.
    Returns X_train, X_val, y_train, y_val, feature_names
    """
    df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))

    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    feature_names = X.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_val_scaled, y_train, y_val, feature_names


def load_scaler():
    return joblib.load(SCALER_PATH)
