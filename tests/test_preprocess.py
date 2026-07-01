import pandas as pd
from src.preprocess import split_features_target, build_preprocessor, FEATURE_COLUMNS


def test_split_features_target():
    df = pd.DataFrame([{**{c: 1 for c in FEATURE_COLUMNS}, "target": 0}])
    x, y = split_features_target(df)
    assert list(x.columns) == FEATURE_COLUMNS
    assert y.iloc[0] == 0


def test_preprocessor_fit_transform():
    df = pd.DataFrame([
        {**{c: 1 for c in FEATURE_COLUMNS}, "target": 0},
        {**{c: 2 for c in FEATURE_COLUMNS}, "target": 1},
    ])
    x, _ = split_features_target(df)
    transformed = build_preprocessor().fit_transform(x)
    assert transformed.shape[0] == 2
