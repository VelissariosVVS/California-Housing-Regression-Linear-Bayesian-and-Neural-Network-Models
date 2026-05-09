from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def train_bayesian_regression(X_train, y_train):

    model = BayesianRidge()

    model.fit(X_train, y_train)

    return model


def evaluate_bayesian_regression(model, X, y):

    preds, std = model.predict(X, return_std=True)

    mse = mean_squared_error(y, preds)

    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
        "predictions": preds,
        "uncertainty_std": std
    }