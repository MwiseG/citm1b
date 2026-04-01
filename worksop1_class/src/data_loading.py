"""Data loading helpers for workshop 1."""

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_housing_data(as_frame: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the California housing dataset.

    Parameters
    ----------
    as_frame : bool, default=True
        Kept for API clarity with sklearn; this workshop expects DataFrame output.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and target vector.
    """
    housing = fetch_california_housing(as_frame=as_frame)
    data = housing.data.copy()
    target = housing.target.rename("MedHouseVal")
    return data, target


def add_income_category(data: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Add an `income_cat` column based on quantiles of `MedInc`.

    Parameters
    ----------
    data : pd.DataFrame
        Input feature table that contains the `MedInc` column.
    n_bins : int, default=5
        Number of quantile bins.

    Returns
    -------
    pd.DataFrame
        Copy of input data with extra `income_cat` column.
    """
    df = data.copy()
    df["income_cat"] = pd.qcut(df["MedInc"], q=n_bins, labels=False, duplicates="drop")
    return df
