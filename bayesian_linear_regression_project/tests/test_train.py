from torch.utils.data import DataLoader

from src.preprocessing import prepare_data
from src.datasets import create_tensor_dataset
from src.mlp import HousingNN
from src.train import train_model
from src.evaluation import evaluate_model, get_predictions
from src.plots import (
    plot_loss_curves,
    plot_predictions_vs_actual,
    plot_residuals,
)

data = prepare_data()

train_dataset = create_tensor_dataset(
    data["X_train_scaled"],
    data["y_train"]
)

val_dataset = create_tensor_dataset(
    data["X_val_scaled"],
    data["y_val"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

model = HousingNN(input_dim=8)

model, history = train_model(
    model,
    train_loader,
    val_loader,
    epochs=200,
    learning_rate=0.001
)

val_metrics = evaluate_model(model, val_loader)

print(val_metrics)

val_preds, val_targets = get_predictions(
    model,
    val_loader
)

plot_loss_curves(history)

plot_predictions_vs_actual(
    val_targets,
    val_preds
)

plot_residuals(
    val_targets,
    val_preds
)