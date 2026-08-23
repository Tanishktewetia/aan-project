"""Train several classifiers, compare them, and save separate graphs."""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from utils import load_and_prepare_data, DATASET_DIR

MODELS_DIR = os.path.join("models", "benchmarks")
RESULTS_DIR = os.path.join("results", "model_comparison")
PREDICTIONS_DIR = os.path.join("results", "predictions")


def model_factories():
    """
    Notes on tuning (see README for the full explanation):
    - This dataset has a very strong linear signal (ram correlates ~0.92 with
      price_range), so linear models (logistic regression, linear-kernel SVM)
      outperform tree-based models and even the plain ANN here.
    - SVM uses a linear kernel with a higher C (less regularization) - this
      was found by testing C in [0.1, 0.5, 1, 2, 5, 10] and picking the best.
    - Logistic Regression's C was tuned the same way.
    - ANN uses tanh activation, which slightly beat relu on this dataset.
    - voting_ensemble combines the three strongest models (soft voting =
      averages predicted probabilities) to squeeze out a bit more accuracy.
    """
    return {
        "ann": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="tanh",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000, C=10, random_state=42
        ),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=8, random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "svm": lambda: SVC(kernel="linear", C=10, probability=True, random_state=42),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5),
        "voting_ensemble": lambda: VotingClassifier(
            estimators=[
                ("svm", SVC(kernel="linear", C=10, probability=True, random_state=42)),
                ("logistic_regression", LogisticRegression(max_iter=2000, C=10, random_state=42)),
                ("ann", MLPClassifier(
                    hidden_layer_sizes=(64, 32), activation="tanh",
                    max_iter=1000, random_state=42,
                )),
            ],
            voting="soft",
        ),
    }


def save_graphs(model_name, y_true, y_pred, scores):
    graph_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(graph_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted price_range")
    plt.ylabel("Actual price_range")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "confusion_matrix.png"))
    plt.close()

    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [scores[label.lower().replace("-", "_")] for label in labels]
    plt.figure(figsize=(6, 5))
    sns.barplot(x=labels, y=values, hue=labels, palette="viridis", legend=False)
    plt.ylim(0, 1)
    plt.title(f"Metrics - {model_name}")
    for index, value in enumerate(values):
        plt.text(index, value + 0.01, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "metrics.png"))
    plt.close()


def save_test_predictions(model_name, model, scaler):
    test_df = pd.read_csv(os.path.join("dataset", "test.csv"))
    ids = test_df["id"] if "id" in test_df.columns else range(len(test_df))
    features = test_df.drop(columns=["id"]) if "id" in test_df.columns else test_df
    predictions = model.predict(scaler.transform(features))
    prediction_dir = os.path.join(PREDICTIONS_DIR, model_name)
    os.makedirs(prediction_dir, exist_ok=True)
    pd.DataFrame({
        "id": ids,
        "predicted_price_range": predictions,
    }).to_csv(os.path.join(prediction_dir, "predictions.csv"), index=False)


def save_comparison_graphs(metrics, confusion_matrices):
    """Save a visual report covering every model and evaluation metric."""
    metric_columns = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    display_names = metrics["model"].str.replace("_", " ").str.title()

    figure, axes = plt.subplots(2, 2, figsize=(18, 13))

    metric_values = metrics[metric_columns].to_numpy().T
    x_positions = range(len(metrics))
    bar_width = 0.2
    for index, (label, values) in enumerate(zip(metric_labels, metric_values)):
        offsets = [position + (index - 1.5) * bar_width for position in x_positions]
        axes[0, 0].bar(offsets, values, width=bar_width, label=label)
    axes[0, 0].set_xticks(list(x_positions), display_names, rotation=30, ha="right")
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Validation performance by model")
    axes[0, 0].legend(ncol=2)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].errorbar(
        display_names,
        metrics["cv_accuracy_mean"],
        yerr=metrics["cv_accuracy_std"],
        fmt="o",
        capsize=5,
        color="#2a6fbb",
    )
    axes[0, 1].set_ylim(0, 1.08)
    axes[0, 1].set_ylabel("5-fold accuracy")
    axes[0, 1].set_title("Cross-validation accuracy and variability")
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].grid(axis="y", alpha=0.25)

    heatmap_data = metrics.set_index("model")[metric_columns + ["cv_accuracy_mean"]]
    heatmap_data.columns = metric_labels + ["CV accuracy"]
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=axes[1, 0],
        cbar_kws={"label": "Score"},
    )
    axes[1, 0].set_title("All model scores")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")

    mean_cm = sum(confusion_matrices.values())
    sns.heatmap(
        mean_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        cbar=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Combined validation predictions (all models)")
    axes[1, 1].set_xlabel("Predicted price_range")
    axes[1, 1].set_ylabel("Actual price_range")

    figure.suptitle("Complete Model Performance Dashboard", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(os.path.join(RESULTS_DIR, "performance_dashboard.png"), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, (model_name, matrix) in zip(axes.flat, confusion_matrices.items()):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(model_name.replace("_", " ").title())
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    for axis in axes.flat[len(confusion_matrices):]:
        axis.axis("off")
    figure.suptitle("Validation Confusion Matrices for Every Model", fontsize=18, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(os.path.join(RESULTS_DIR, "confusion_matrices_comparison.png"), dpi=180)
    plt.close(figure)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_val, y_train, y_val, _ = load_and_prepare_data()
    scaler = joblib.load(os.path.join("models", "preprocessing", "scaler.pkl"))

    # Full (unscaled) training data, used only for 5-fold cross-validation
    # below, so results aren't just a single lucky train/val split.
    full_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    X_full = full_df.drop("price_range", axis=1)
    y_full = full_df["price_range"]

    rows = []
    confusion_matrices = {}

    for model_name, create_model in model_factories().items():
        print(f"\nTraining {model_name}...")
        model = create_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="macro"),
            "recall": recall_score(y_val, y_pred, average="macro"),
            "f1_score": f1_score(y_val, y_pred, average="macro"),
        }

        # 5-fold cross-validation (scaler + model in a pipeline, fit fresh
        # each fold, so there's no data leakage). This checks the score is
        # stable and not just a lucky split.
        cv_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", create_model()),
        ])
        cv_scores = cross_val_score(cv_pipeline, X_full, y_full, cv=5)
        scores["cv_accuracy_mean"] = cv_scores.mean()
        scores["cv_accuracy_std"] = cv_scores.std()

        rows.append({"model": model_name, **scores})
        confusion_matrices[model_name] = confusion_matrix(y_val, y_pred)
        model_dir = os.path.join(MODELS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        save_graphs(model_name, y_val, y_pred, scores)
        save_test_predictions(model_name, model, scaler)
        print(", ".join(f"{key}={value:.4f}" for key, value in scores.items()))

    metrics = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)

    save_comparison_graphs(metrics, confusion_matrices)

    print("\n--- Model comparison ---")
    print(metrics.to_string(index=False))
    print(f"\nMetrics and graphs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()