import torch
import torch.nn as nn

class HousingNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)