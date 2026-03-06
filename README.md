# Physics-Informed Neural Network for the Heat Equation

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **1D Heat Equation**.  
Instead of relying purely on classical numerical solvers, the neural network learns the solution by embedding the **governing physical equation directly into the loss function**.

The model learns the function **u(x,t)** while satisfying:

- the **heat equation (PDE)**
- the **initial condition**
- the **boundary conditions**

PINNs combine **deep learning and physics constraints** to approximate continuous solutions of differential equations.

---

# Problem Description

We solve the **1D Heat Equation**

```
∂u/∂t = ∂²u/∂x²
```

Domain:

```
x ∈ [0,1]
t ∈ [0,1]
```

Boundary Conditions:

```
u(0,t) = 0
u(1,t) = 0
```

Initial Condition:

```
u(x,0) = sin(πx)
```

The **analytical solution** is:

```
u(x,t) = exp(-π²t) sin(πx)
```

---

# Project Structure

```
pinn-heat-equation/
│
├── pinn_model.pt            # Trained PINN model
│
├── src/
│   ├── train.py             # Training script for the PINN
│   └── model.py             # Neural network architecture
│
├── notebooks/
│   └── heat_equation.ipynb  # Evaluation & visualization notebook
│
├── results/                 # Generated plots
│
├── requirements.txt
└── README.md
```

---

# Physics-Informed Neural Networks

Unlike traditional neural networks trained purely on labeled data, **PINNs incorporate physical laws directly into the loss function**.

The loss consists of three main components.

---

## 1. PDE Residual Loss

The network must satisfy the heat equation:

```
L_pde = MSE( u_t − u_xx )
```

Derivatives are computed using **PyTorch automatic differentiation**.

---

## 2. Boundary Condition Loss

Ensures the solution respects:

```
u(0,t) = 0
u(1,t) = 0
```

---

## 3. Initial Condition Loss

Ensures the model matches the initial temperature distribution:

```
u(x,0) = sin(πx)
```

---

## Total Loss

```
L_total = L_pde + L_boundary + L_initial
```

---

# Neural Network Architecture

The PINN is implemented as a **fully connected feed-forward neural network**.

Input:

```
(x, t)
```

Output:

```
u(x,t)
```

Architecture used in this project:

```
2 → 50 → 50 → 50 → 1
```

Activation function:

```
Tanh
```

Smooth activation functions are important because **the model must compute derivatives of the output**.

---

# Training Procedure

Training is performed by sampling **collocation points** across the domain.

Training pipeline:

1. Randomly sample points in space-time
2. Compute network prediction
3. Compute derivatives using automatic differentiation
4. Evaluate PDE residual
5. Minimize the total loss

Optimizer used:

```
Adam
```

Training runs for several thousand iterations.

---

# Visualization

After training, we compare:

- Analytical solution
- PINN predicted solution
- Error distribution

These are visualized as **contour plots over the space–time domain**.

---

# Analytical Solution


```
![Analytical Solution](/Users/bardia/Desktop/pinn-heat-equation/notebooks/results/analytical_solution.png)
```

---

# PINN Predicted Solution

```
![PINN Prediction](/Users/bardia/Desktop/pinn-heat-equation/notebooks/results/pinn_solution.png)
```

---

# Error Distribution

```
![Prediction Error](/Users/bardia/Desktop/pinn-heat-equation/notebooks/results/error_heatmap.png)
```

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/pinn-heat-equation.git
cd pinn-heat-equation
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Train the model:

```bash
python src/train.py
```

This will:

- train the PINN
- generate the solution plot
- save the figure in:

```
results/pinn_solution.png
```

---

# Using the Notebook

Open the notebook:

```
notebooks/heat_equation.ipynb
```

The notebook allows you to:

- load the trained model
- generate analytical solution
- compare predictions
- visualize error maps

---

# Future Improvements

Possible extensions of this project:

- Solve the **2D heat equation**
- Implement **adaptive collocation sampling**
- Add **L-BFGS optimizer for improved convergence**
- Extend PINNs to **Burgers' equation**
- Apply PINNs to **Navier-Stokes equations**
- Add **Fourier feature embeddings**

---

# References

Raissi, M., Perdikaris, P., Karniadakis, G. (2019)

**Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations**

https://arxiv.org/abs/1711.10561
