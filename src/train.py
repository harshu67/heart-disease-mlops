"""Train, evaluate and track heart disease classification models."""
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

try:
    from src.data_acquisition import download_dataset
    from src.preprocess import build_preprocessor, split_features_target
except ModuleNotFoundError:
    from data_acquisition import download_dataset
    from preprocess import build_preprocessor, split_features_target


def load_data(path: str = "data/raw/heart_disease_uci.csv") -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        return download_dataset(str(data_path))
    return pd.read_csv(data_path)


def evaluate_model(model, x_test, y_test):
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def save_plots(model, x_test, y_test, output_dir="models/artifacts"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax)
    cm_path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
    roc_path = Path(output_dir) / "roc_curve.png"
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    return cm_path, roc_path


def main():
    df = load_data()
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            {"model__C": [0.1, 1.0, 10.0], "model__solver": ["liblinear"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"model__n_estimators": [100, 200], "model__max_depth": [3, 5, None]},
        ),
    }

    mlflow.set_experiment("heart-disease-uci-mlops")
    results = []
    best_score = -1
    best_model = None
    best_name = None

    for name, (estimator, params) in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])
        grid = GridSearchCV(pipeline, params, scoring="roc_auc", cv=5, n_jobs=-1)
        with mlflow.start_run(run_name=name):
            grid.fit(x_train, y_train)
            metrics = evaluate_model(grid.best_estimator_, x_test, y_test)
            cv_auc = cross_val_score(grid.best_estimator_, x, y, cv=5, scoring="roc_auc").mean()
            metrics["cv_roc_auc"] = cv_auc
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            cm_path, roc_path = save_plots(grid.best_estimator_, x_test, y_test)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")
            row = {"model": name, "best_params": grid.best_params_, **metrics}
            results.append(row)
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = grid.best_estimator_
                best_name = name

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/heart_pipeline.joblib")
    with open("models/model_metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    print(f"Best model: {best_name}, ROC-AUC: {best_score:.4f}")


if __name__ == "__main__":
    main()
