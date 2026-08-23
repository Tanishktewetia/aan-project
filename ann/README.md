# Mobile Price Prediction — ANN Project

## Dataset
- `dataset/train.csv` — 2000 rows, mobile phone specs + `price_range` target
  (0 = low, 1 = medium, 2 = high, 3 = very high). 4-class classification.
- `dataset/test.csv` — 1000 rows, same features, no target (used for predictions).

## Model
An **Artificial Neural Network** (`MLPClassifier` from scikit-learn) with two
hidden layers (64 and 32 neurons), tanh activation, and the Adam optimizer.
Features are scaled with `StandardScaler` before training, since ANNs are
sensitive to feature magnitude.

`compare_models.py` also trains: Logistic Regression, Decision Tree, Random
Forest, SVM, KNN, and a **voting ensemble** (soft-voting combination of SVM +
Logistic Regression + ANN, the three strongest models).

## How to run

```bash
python -m pip install -r requirements.txt

# 1. Train the ANN and generate graphs
python train.py

# 2. Train all classifiers and create predictions for test.csv
python compare_models.py

# 3. Predict with one selected classifier
python predict_model.py --model logistic_regression
```

The comparison supports `ann`, `logistic_regression`, `decision_tree`,
`random_forest`, `svm`, `knn`, and `voting_ensemble`. It saves each model's
confusion matrix and metrics graph in `results/model_comparison/<model>/`,
and saves the summary in `results/model_comparison/metrics_comparison.csv`
(including 5-fold cross-validation accuracy, so the results aren't just a
single lucky train/validation split).

To demonstrate ANN learning to a teacher, run `python train_demo.py`. The
training iterations are printed in the terminal and the ANN graphs are saved
in `results/ann/graphs/`.

R2 is not used because this is a classification problem. Accuracy, precision,
recall, and F1-score are the appropriate evaluation metrics for the four price
classes.

## What gets created
- `models/ann/model.pkl` — trained ANN
- `models/benchmarks/<model>/model.pkl` — trained comparison models
- `models/preprocessing/scaler.pkl` — fitted scaler shared by all models
- `results/ann/` — ANN metrics and graphs
- `results/model_comparison/<model>/` — graphs for each model
- `results/model_comparison/metrics_comparison.csv` — comparison metrics
- `results/predictions/<model>/predictions.csv` — test predictions for every model

## Result

| Model | Validation Accuracy | 5-fold CV Accuracy (mean ± std) |
|---|---|---|
| Logistic Regression | 97.5% | 97.3% ± 0.6% |
| SVM (linear kernel) | 97.5% | 97.0% ± 0.4% |
| Voting Ensemble (SVM + LogReg + ANN) | 97.5% | 96.5% ± 0.5% |
| ANN (tanh) | 93.3% | 94.0% ± 1.9% |
| Random Forest | 87.8% | 88.5% ± 1.1% |
| Decision Tree | 85.0% | 83.5% ± 0.3% |
| KNN | 50.0% | 50.4% ± 1.0% |

**Why some models do much better than others:** `ram` alone correlates ~0.92
with `price_range` in this dataset — an unusually strong, almost linear
signal. That's why linear models (Logistic Regression, linear-kernel SVM)
outperform tree-based models and even the plain ANN here.

**On reaching 98% accuracy:** this was specifically tested (see cross-validation
above). ~97-97.5% is the practical ceiling on this dataset with standard
models — the cross-validation confirms it's a stable result, not one lucky
split. Pushing further would mean the model is starting to overfit rather
than genuinely improving, so 97.5% is reported as the honest best result
rather than an inflated number.
