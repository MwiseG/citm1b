import numpy as np

from src.data_loading import load_housing_data
from src.preprocessing import build_preprocessing_pipeline, stratified_split


def test_stratified_split_shapes():
    data, target = load_housing_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(data, target)

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_preprocessing_no_nan():
    data, target = load_housing_data()
    X_train, _, _, _, _, _ = stratified_split(data, target)

    numeric_features = list(X_train.columns)
    preprocessor = build_preprocessing_pipeline(numeric_features)
    X_train_prepared = preprocessor.fit_transform(X_train)

    assert not np.isnan(X_train_prepared).any()
