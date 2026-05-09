import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_predictions(model, data_loader, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            y_pred = model(X_batch).cpu().numpy()

            preds.append(y_pred)
            targets.append(y_batch.numpy())

    preds = np.vstack(preds).ravel()
    targets = np.vstack(targets).ravel()

    return preds, targets

def evaluate_model(model, data_loader, device=None):
    preds, targets = get_predictions(model, data_loader, device)

    mse = mean_squared_error(targets, preds)

    return{
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(targets, preds),
        'r2': r2_score(targets, preds)
    }