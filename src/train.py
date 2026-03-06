import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===== Device =====
device = torch.device("cpu")

# ===== Neural Network =====
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.Tanh(),
            nn.Linear(50,1)
        )

    def forward(self,x,t):
        inputs = torch.cat([x,t],dim=1)
        return self.network(inputs)


# ===== Loss Functions =====
def physics_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True

    u = model(x,t)

    u_t = torch.autograd.grad(u, t,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    u_x = torch.autograd.grad(u, x,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]

    return torch.mean((u_t - u_xx)**2)


def boundary_loss(model, t):
    x0 = torch.zeros_like(t)
    x1 = torch.ones_like(t)

    u0 = model(x0,t)
    u1 = model(x1,t)

    return torch.mean(u0**2) + torch.mean(u1**2)


def initial_loss(model, x):
    t0 = torch.zeros_like(x)
    u_pred = model(x,t0)
    u_true = torch.sin(np.pi*x)
    return torch.mean((u_pred - u_true)**2)


# ===== Training Function =====
def train(model, epochs=5000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        x = torch.rand(1000,1)
        t = torch.rand(1000,1)

        loss_pde = physics_loss(model, x, t)

        t_b = torch.rand(200,1)
        loss_bc = boundary_loss(model, t_b)

        x_i = torch.rand(200,1)
        loss_ic = initial_loss(model, x_i)

        loss = loss_pde + loss_bc + loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


# ===== Plotting Function =====
def plot_solution(model):
    # Ensure results folder exists
    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    x = np.linspace(0,1,100)
    t = np.linspace(0,1,100)
    X, T = np.meshgrid(x,t)

    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1,1)
    t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1,1)

    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).numpy()

    U = u_pred.reshape(100,100)

    # PINN Solution
    plt.figure(figsize=(8,6))
    plt.imshow(U, extent=[0,1,0,1], origin="lower", aspect="auto")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("PINN Solution")
    plt.savefig(RESULTS_DIR / "pinn_solution.png")
    plt.show()


# ===== Script execution =====
if __name__ == "__main__":
    model = PINN().to(device)
    model = train(model)
    torch.save(model, Path(__file__).resolve().parent.parent / "pinn_model.pt")
    plot_solution(model)