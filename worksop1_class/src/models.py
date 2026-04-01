"""Baseline models for workshop 1."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_baseline_linear_regression(X_train, y_train):
    """Fit and return a simple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def compute_rmse(model, X, y) -> float:
    """Compute root mean squared error (RMSE)."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return float(np.sqrt(mse))
