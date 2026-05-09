import matplotlib.pyplot as plt

def plot_loss_curves(history):
    plt.figure(figsize=(8, 5))

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_predictions_vs_actual(y_true, y_pred):
    plt.figure(figsize=(6, 6))

    plt.scatter(y_true, y_pred, alpha=0.4)

    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title("Predicted vs Actual Values")

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )

    plt.grid(True)
    plt.show()


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted Median House Value")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.grid(True)
    plt.show()