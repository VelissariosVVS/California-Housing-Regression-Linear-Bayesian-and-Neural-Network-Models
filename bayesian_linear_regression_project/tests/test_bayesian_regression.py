from src.preprocessing import prepare_data

from src.bayesian_regression import (
    train_bayesian_regression,
    evaluate_bayesian_regression
)

from src.plots import (
    plot_predictions_vs_actual,
    plot_residuals
)


data = prepare_data()

model = train_bayesian_regression(
    data["X_train_scaled"],
    data["y_train"]
)

val_results = evaluate_bayesian_regression(
    model,
    data["X_val_scaled"],
    data["y_val"]
)

print("Validation metrics:")
print({
    "mse": val_results["mse"],
    "rmse": val_results["rmse"],
    "mae": val_results["mae"],
    "r2": val_results["r2"],
})

plot_predictions_vs_actual(
    data["y_val"],
    val_results["predictions"]
)

plot_residuals(
    data["y_val"],
    val_results["predictions"]
)
