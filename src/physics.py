import torch

def heat_residual(model, x, t, alpha=1.0):
    """
    Computes the PDE residual: u_t - alpha * u_xx
    """
    # Enable gradient tracking
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)

    # First derivative wrt t
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # First derivative wrt x
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Second derivative wrt x
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # Residual of PDE
    return u_t - alpha * u_xx