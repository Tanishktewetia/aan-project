"""
train.py
Trains an Artificial Neural Network (MLPClassifier) on the mobile price
dataset, evaluates it, saves the model, and saves graphs + metrics.

Run:  python train.py
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from utils import load_and_prepare_data

MODEL_DIR = os.path.join("models", "ann")
RESULTS_DIR = os.path.join("results", "ann")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    print("Loading and preparing data...")
    X_train, X_val, y_train, y_val, feature_names = load_and_prepare_data()

    print("Training the ANN (MLPClassifier)...")
    # activation="tanh" was found (via compare_models.py testing) to slightly
    # outperform relu on this dataset.
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="tanh",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="macro")
    rec = recall_score(y_val, y_pred, average="macro")
    f1 = f1_score(y_val, y_pred, average="macro")

    print("\n--- Validation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    # --- Save model ---
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    print(f"\nModel saved to {MODEL_DIR}/model.pkl")

    # --- Save metrics to CSV ---
    metrics_df = pd.DataFrame([{
        "model": "ANN (MLPClassifier)",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    # --- Graph 1: Confusion matrix ---
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted price_range")
    plt.ylabel("Actual price_range")
    plt.title("Confusion Matrix - ANN")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "confusion_matrix.png"))
    plt.close()

    # --- Graph 2: Loss curve (training loss per iteration) ---
    plt.figure(figsize=(6, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve - ANN")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "loss_curve.png"))
    plt.close()

    # --- Graph 3: Accuracy/Precision/Recall/F1 bar chart ---
    plt.figure(figsize=(6, 5))
    scores = [acc, prec, rec, f1]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    sns.barplot(x=labels, y=scores, hue=labels, palette="viridis", legend=False)
    plt.ylim(0, 1)
    plt.title("ANN Performance Metrics")
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "metrics_summary.png"))
    plt.close()

    print(f"\nGraphs saved to {GRAPHS_DIR}/")
    print("Done. Run 'python compare_models.py' to train all models and create test predictions.")


if __name__ == "__main__":
    main()
