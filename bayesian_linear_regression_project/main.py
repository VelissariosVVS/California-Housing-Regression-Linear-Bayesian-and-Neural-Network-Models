import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.preprocessing import prepare_data
from src.datasets import create_tensor_dataset

from src.sklearn_linear_regression import (
    train_linear_regression,
    evaluate_linear_regression,
)

from src.bayesian_regression import (
    train_bayesian_regression,
    evaluate_bayesian_regression,
)

from src.mlp import HousingNN
from src.train import train_model
from src.evaluation import evaluate_model, get_predictions


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    data = prepare_data()

    results = {}

    # -------------------------
    # Linear Regression
    # -------------------------
    linear_model = train_linear_regression(
        data["X_train_scaled"],
        data["y_train"],
    )

    linear_test = evaluate_linear_regression(
        linear_model,
        data["X_test_scaled"],
        data["y_test"],
    )

    results["Linear Regression"] = {
        "mse": linear_test["mse"],
        "rmse": linear_test["rmse"],
        "mae": linear_test["mae"],
        "r2": linear_test["r2"],
    }

    # -------------------------
    # Bayesian Ridge Regression
    # -------------------------
    bayesian_model = train_bayesian_regression(
        data["X_train_scaled"],
        data["y_train"],
    )

    bayesian_test = evaluate_bayesian_regression(
        bayesian_model,
        data["X_test_scaled"],
        data["y_test"],
    )

    results["Bayesian Ridge Regression"] = {
        "mse": bayesian_test["mse"],
        "rmse": bayesian_test["rmse"],
        "mae": bayesian_test["mae"],
        "r2": bayesian_test["r2"],
    }

    # -------------------------
    # MLP
    # -------------------------
    train_dataset = create_tensor_dataset(
        data["X_train_scaled"],
        data["y_train"],
    )

    val_dataset = create_tensor_dataset(
        data["X_val_scaled"],
        data["y_val"],
    )

    test_dataset = create_tensor_dataset(
        data["X_test_scaled"],
        data["y_test"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    mlp_model = HousingNN(input_dim=8)

    mlp_model, history = train_model(
        mlp_model,
        train_loader,
        val_loader,
        epochs=200,
        learning_rate=0.001,
    )

    mlp_test = evaluate_model(
        mlp_model,
        test_loader,
    )

    results["MLP"] = mlp_test

    # -------------------------
    # Save results
    # -------------------------
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nFinal Test Results")
    print("------------------")

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE : {metrics['mae']:.4f}")
        print(f"R²  : {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()