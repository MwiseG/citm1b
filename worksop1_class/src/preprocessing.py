"""Preprocessing utilities for workshop 1."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loading import add_income_category


def stratified_split(
    data: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/val/test with stratification on income category.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = add_income_category(data)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_val_idx, test_idx in splitter.split(df, df["income_cat"]):
        train_val_data = df.iloc[train_val_idx].drop(columns=["income_cat"])
        test_data = df.iloc[test_idx].drop(columns=["income_cat"])
        y_train_val = target.iloc[train_val_idx]
        y_test = target.iloc[test_idx]

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    df_train_val = add_income_category(train_val_data)
    for train_idx, val_idx in splitter2.split(df_train_val, df_train_val["income_cat"]):
        X_train = df_train_val.iloc[train_idx].drop(columns=["income_cat"])
        X_val = df_train_val.iloc[val_idx].drop(columns=["income_cat"])
        y_train = y_train_val.iloc[train_idx]
        y_val = y_train_val.iloc[val_idx]

    X_test = test_data
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessing_pipeline(numeric_features: list) -> ColumnTransformer:
    """
    Build a numeric preprocessing pipeline:
    - median imputation
    - standard scaling
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ]
    )
