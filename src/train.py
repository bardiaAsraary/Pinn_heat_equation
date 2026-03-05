import torch
import numpy as np
from model import PINN
from physics import heat_residual

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
alpha = 1.0
epochs = 3000
lr = 0.001
N_train = 1000

# Initialize model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Domain: x in [0,1], t in [0,1]
x_train = torch.rand(N_train, 1)
t_train = torch.rand(N_train, 1)

# Boundary points: x=0 and x=1
t_bc = torch.rand(N_train,1)
x0 = torch.zeros(N_train,1)
x1 = torch.ones(N_train,1)

# Initial condition: t=0
x_ic = torch.rand(N_train,1)
t_ic = torch.zeros(N_train,1)
u_ic_target = torch.sin(np.pi * x_ic)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # PDE residual
    res = heat_residual(model, x_train, t_train)
    physics_loss = torch.mean(res**2)

    # Boundary condition loss
    u0 = model(x0, t_bc)
    u1 = model(x1, t_bc)
    bc_loss = torch.mean(u0**2) + torch.mean(u1**2)

    # Initial condition loss
    u_ic = model(x_ic, t_ic)
    ic_loss = torch.mean((u_ic - u_ic_target)**2)

    # Total loss
    loss = physics_loss + bc_loss + ic_loss

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(model, "pinn_model.pt")
print("Model saved as pinn_model.pt")