"""Predict test.csv with a selected model from compare_models.py."""

import argparse
import os

import joblib
import pandas as pd

from compare_models import model_factories, MODELS_DIR
from utils import DATASET_DIR, load_scaler


def main():
    choices = list(model_factories())
    parser = argparse.ArgumentParser(description="Predict with a selected classifier")
    parser.add_argument("--model", choices=choices, required=True)
    args = parser.parse_args()

    model_path = os.path.join(MODELS_DIR, args.model, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} does not exist. Run 'python compare_models.py' first."
        )

    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))
    ids = test_df["id"] if "id" in test_df.columns else range(len(test_df))
    X_test = test_df.drop(columns=["id"]) if "id" in test_df.columns else test_df
    predictions = joblib.load(model_path).predict(load_scaler().transform(X_test))

    output_dir = os.path.join("results", "predictions", args.model)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame({"id": ids, "predicted_price_range": predictions}).to_csv(
        output_path, index=False
    )
    print(f"Predictions saved to {output_path}")
    print(pd.Series(predictions).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()