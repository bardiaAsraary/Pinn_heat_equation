import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network
    Input: x, t
    Output: u(x,t)
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),  # 2 inputs: x and t
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)   # 1 output: u(x,t)
        )

    def forward(self, x, t):
        # Concatenate x and t as the input
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)