import torch
from torch.utils.data import TensorDataset

def create_tensor_dataset(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(
        y.values if hasattr(y, 'values') else y,
        dtype=torch.float32
    ).view(-1, 1)

    return TensorDataset(X_tensor, y_tensor)