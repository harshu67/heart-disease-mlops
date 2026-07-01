"""Download and prepare the Heart Disease UCI dataset.

The file is downloaded from the UCI Machine Learning Repository mirror used by many
examples. Missing values in this dataset are represented as '?'.
"""
from pathlib import Path
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]


def download_dataset(raw_path: str = "data/raw/heart_disease_uci.csv") -> pd.DataFrame:
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(raw_path, index=False)
    return df


if __name__ == "__main__":
    data = download_dataset()
    print(f"Downloaded dataset with shape: {data.shape}")
    print(data.head())
