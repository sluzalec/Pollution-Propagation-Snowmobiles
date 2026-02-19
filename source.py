from token import NUMBER
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

import time
import os
from typing import Tuple
import math
import imageio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

LENGTH = 1.                     # Domain size in x axis. Always starts at 0
TOTAL_TIME = .6               # Domain size in t axis. Always starts at 0
N_POINTS = 25                   # Number of in single axis
N_POINTS_PLOT = 150             # Number of points in single axis used in plotting
WEIGHT_RESIDUAL = 20.0          # Weight of residual part of loss function
WEIGHT_INITIAL = 200.0            # Weight of initial part of loss function
WEIGHT_BOUNDARY = 10.0          # Weight of boundary part of loss function
LAYERS = 2
NEURONS_PER_LAYER = 100
EPOCHS = 20_000
LEARNING_RATE = 0.002

## Initial condition

def initial_condition(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)

## PINN


class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

        self.layer_in = nn.Linear(3, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)
        self.layer_out.bias.data.fill_(0) # Initialize the bias of the output layer to zero to help the function start from zero (as this is our initial condition)


        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
    def forward(self, x, y, t):

        x_stack = torch.cat([x, y, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits

    def device(self):
        return next(self.parameters()).device

def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y, t)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdt(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, y, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, y, t)
    return df(f_value, x, order=order)

def dfdy(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, order: int = 1):
    f_value = f(pinn, x, y, t)
    return df(f_value, y, order=order)

def get_boundary_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    """
         .+------+
       .' |    .'|
      +---+--+'  |
      |   |  |   |
    y |  ,+--+---+
      |.'    | .' t
      +------+'
         x
    """
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    t_linspace = torch.linspace(t_domain[0], t_domain[1], n_points)

    x_grid, t_grid = torch.meshgrid( x_linspace, t_linspace, indexing="ij")
    y_grid, _      = torch.meshgrid( y_linspace, t_linspace, indexing="ij")

    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t_grid = t_grid.reshape(-1, 1).to(device)
    t_grid.requires_grad = requires_grad

    x0 = torch.full_like(t_grid, x_domain[0], requires_grad=requires_grad)
    x1 = torch.full_like(t_grid, x_domain[1], requires_grad=requires_grad)
    y0 = torch.full_like(t_grid, y_domain[0], requires_grad=requires_grad)
    y1 = torch.full_like(t_grid, y_domain[1], requires_grad=requires_grad)

    down    = (x_grid, y0,     t_grid)
    up      = (x_grid, y1,     t_grid)
    left    = (x0,     y_grid, t_grid)
    right   = (x1,     y_grid, t_grid)

    return down, up, left, right

def get_initial_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    x_grid, y_grid = torch.meshgrid( x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)

def get_interior_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points, requires_grad=requires_grad)
    y_raw = torch.linspace(y_domain[0], y_domain[1], steps=n_points, requires_grad=requires_grad)
    t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points, requires_grad=requires_grad)
    grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

    x = grids[0].reshape(-1, 1).to(device)
    y = grids[1].reshape(-1, 1).to(device)
    t = grids[2].reshape(-1, 1).to(device)

    return x, y, t


G = torch.eye(N_POINTS * N_POINTS * N_POINTS)
def linearized(ix, iy, it):
  return ix * N_POINTS * N_POINTS + iy * N_POINTS + it

def nearby(ix, iy, it):
  return [(ix + 1, iy, it), (ix - 1, iy, it), (ix, iy - 1, it), (ix, iy + 1, it), (ix, iy, it -1), (ix, iy, it+1)]

for ix in range(N_POINTS):
  for iy in range(N_POINTS):
    for it in range(N_POINTS):
      i = linearized(ix, iy, it)
      G[i, i] = 1
for ix in range(1, N_POINTS - 1):
  for iy in range(1, N_POINTS - 1):
    for it in range(1, N_POINTS - 1):
      i = linearized(ix, iy, it)
      G[i, i] = 6
      for jx, jy, jt in nearby(ix, iy, it):
        j = linearized(jx, jy, jt)
        G[i, j] = -1
hx = 1.0 / N_POINTS
hy = 1.0 / N_POINTS
ht = 1.0 / N_POINTS
G = G / (hx*hy*ht)
G = G.to(device)
G_LU = torch.linalg.lu_factor(G)


class Loss:
    def __init__(
        self,
        x_domain: Tuple[float, float],
        y_domain: Tuple[float, float],
        t_domain: Tuple[float, float],
        n_points: int,
        initial_condition: Callable,
        weight_r: float = 1.0,
        weight_b: float = 1.0,
        weight_i: float = 1.0,
        verbose: bool = False,
    ):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.n_points = n_points
        self.initial_condition = initial_condition
        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i
        self.Kx=1.0
        self.Ky=0.01

    def dTy(self,y,t):
      #Day (decreases in vertical direction)
      res = torch.where(t < 0.5, 2, 2)
      res = torch.where(y < 0.1, 0, res)
      return res.to(device) * 2.0

    def cannon_x(self,x,y,t):
      return 0.0

    def cannon_y(self,x,y,t):
      return 0.0

    def source(self,y,t):
      res = 10*torch.ones_like(t)
      res = torch.where(y <= 0.2, res, 0)
      res = torch.where(t <= 0.04, res, 0)
      return res

    def _moving_source(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the source term based on a moving source equation.

        This implementation models a wave-like front moving in the positive x-direction
        with a Gaussian profile in the y-direction. It assumes that the spatial
        coordinate grids `self.x` and `self.y` are available as attributes of the class.

        Args:
        x (float or np.ndarray): X-coordinate(s).
        y (float or np.ndarray): Y-coordinate(s).
        t (float): Current time of the simulation.
        H_MAX (float): The absolute maximum height of the wave-like front (at y=0). This can be interpereted as strength of pollution.
        VELOCITY (float): The velocity of the wave front in the x-direction or how fast scooter users are moving.
        WAVE_WIDTH (float): The width of the wave's rising front (transition zone). This is like a gradien, or fall off decay thoruhgout the wave in y-possitive direction.
        Y_SPREAD (float): Controls the spread/width of the wave in the y-direction. This controls how far up the pollution goes.

        Note: This source term depends on spatial coordinates (self.x, self.y)
        and time (t), but not on the current state of the system (the input `y`).
        """

        H_MAX = 0.5
        VELOCITY = 10.0
        WAVE_WIDTH = .8
        Y_SPREAD = 0.4
        # The state `y` is ignored by this source term, which is purely a function
        # of space and time.

        # 1. Calculate the y-dependent maximum height using a Gaussian profile.
        H_y = H_MAX * torch.exp(-y**2 / (2 * Y_SPREAD**2))
        # 2. Calculate the position of the wave's leading edge at time t.
        pos = VELOCITY * t

        # 3. Normalize the x-position relative to the width of the wave front.
        alpha = (pos - x) / WAVE_WIDTH

        # 4. Clamp alpha to the [0, 1] range to handle regions ahead of,
        # inside, and behind the wave front.
        alpha_clamped = torch.clamp(alpha, 0.0, 1.0)

        # 5. Use a smooth cosine function (a "smoothstep") for the transition
        # from 0 to 1 across the wave front.
        height_factor = 0.5 * (1 - torch.cos(math.pi * alpha_clamped))

        # 6. The final source value is the y-dependent max height scaled by the factor.
        final_source_value = H_y * height_factor

        final_source_value = torch.where(y <= 0.2, final_source_value, 0)
        # return torch.where(t <= TOTAL_TIME / 2, final_source_value, 0)
        return final_source_value

    def residual_loss(self, pinn: PINN):
        x, y, t = get_interior_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        loss = dfdt(pinn, x, y, t).to(device) + self.dTy(y, t)*dfdy(pinn, x, y, t).to(device) - self.Kx*dfdx(pinn, x, y, t,order=2).to(device) - self.Ky*dfdy(pinn, x, y, t, order=2).to(device) - self._moving_source(x, y, t).to(device)

        Ginv_loss = torch.linalg.lu_solve(*G_LU, loss.reshape(-1, 1))
        loss_val = torch.dot(loss.reshape(-1), Ginv_loss.reshape(-1))
        return loss_val

    def initial_loss(self, pinn: PINN):
        x, y, t = get_initial_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        pinn_init = self.initial_condition(x, y)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = get_boundary_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        x_down,  y_down,  t_down    = down
        x_up,    y_up,    t_up      = up
        x_left,  y_left,  t_left    = left
        x_right, y_right, t_right   = right

        loss_down  = dfdy( pinn, x_down,  y_down,  t_down  )
        loss_up    = dfdy( pinn, x_up,    y_up,    t_up    )
        loss_left  = dfdx( pinn, x_left,  y_left,  t_left  )
        loss_right = dfdx( pinn, x_right, y_right, t_right )

        return loss_down.pow(2).mean()  + \
            loss_up.pow(2).mean()    + \
            loss_left.pow(2).mean()  + \
            loss_right.pow(2).mean()

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function

        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = \
            self.weight_r * residual_loss + \
            self.weight_i * initial_loss + \
            self.weight_b * boundary_loss

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use instance of this class as if it was a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)

    decayRate = 0.9998
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    loss_values = []
    loss_residual_values = []
    loss_initial_values = []
    loss_boundary_values = []
    for epoch in range(max_epochs):

        try:
            loss_total = loss_fn(nn_approximator)
            loss: torch.Tensor = loss_total[0]
            def getLoss():
              return loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch < 30000:
              my_lr_scheduler.step()

            loss_values.append(loss.item())
            loss_residual_values.append(loss_total[1].item())
            loss_initial_values.append(loss_total[2].item())
            loss_boundary_values.append(loss_total[3].item())
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values), np.array(loss_residual_values), np.array(loss_initial_values), np.array(loss_boundary_values)

def plot_solution(pinn: PINN, x: torch.Tensor, t: torch.Tensor, figsize=(8, 6), dpi=100):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)

    def animate(i):

        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f_final = f(pinn, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(), f_final.detach().numpy(), label=f"Time {float(t[i])}"
            )
            ax.set_ylim(-1, 1)
            ax.legend()

    n_frames = t_raw.shape[0]
    return FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=False)

def plot_color(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, cmap="viridis"):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=0)
    fig.colorbar(c, ax=ax)

    return fig

def plot_3D(z: torch.Tensor, x: torch.Tensor, y: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, limit=0.2):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    c = ax.plot_surface(X, Y, Z)

    return fig

def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.GELU()).to(device)


x_domain = [0.0, LENGTH]
y_domain = [0.0, LENGTH]
t_domain = [0.0, TOTAL_TIME]

# train the PINN
loss_fn = Loss(
    x_domain,
    y_domain,
    t_domain,
    N_POINTS,
    initial_condition,
    WEIGHT_RESIDUAL,
    WEIGHT_INITIAL,
    WEIGHT_BOUNDARY
)

pinn_trained, loss_values, loss_residual_values, loss_initial_values, loss_boundary_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)

pinn = pinn.cpu()


# Loss function
output_dir = "output_plots"
animation_dir = os.path.join(output_dir, "animation_frames")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(animation_dir, exist_ok=True)


# --- Loss Calculation and Printing ---
losses = loss_fn.verbose(pinn)
print(f'Total loss: \t{losses[0]:.5f}    ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f}    ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f}    ({losses[2]:.3E})')
print(f'Bondary loss: \t{losses[3]:.5f}    ({losses[3]:.3E})')


# --- Plotting and Saving Loss Figures ---

# Total Loss
average_loss = running_average(loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss)
ax.set_yscale('log')
# MODIFICATION: Save the figure
fig.savefig(os.path.join(output_dir, "loss_total_average.png"))
plt.close(fig) # Close the plot to free up memory

# Residual Loss
average_loss = running_average(loss_residual_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (residual) (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss)
ax.set_yscale('log')

fig.savefig(os.path.join(output_dir, "loss_residual_average.png"))
plt.close(fig)

# Initial Loss
average_loss = running_average(loss_initial_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (initial) (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss)
ax.set_yscale('log')

fig.savefig(os.path.join(output_dir, "loss_initial_average.png"))
plt.close(fig)

# Boundary Loss
average_loss = running_average(loss_boundary_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (boundary) (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss)
ax.set_yscale('log')

fig.savefig(os.path.join(output_dir, "loss_boundary_average.png"))
plt.close(fig)


# --- Plotting and Saving Initial Condition Figures ---

# Get points for plotting
x, y, _ = get_initial_points(x_domain, y_domain, t_domain, N_POINTS_PLOT, requires_grad=False)

# Initial condition - exact
z = initial_condition(x, y)
fig = plot_color(z, x, y, N_POINTS_PLOT, N_POINTS_PLOT, "Initial condition - exact")

fig.savefig(os.path.join(output_dir, "initial_condition_exact.png"))
plt.close(fig)

# Initial condition - PINN prediction at t=0
t_value = 0.0
t = torch.full_like(x, t_value)
z = pinn(x, y, t)
fig = plot_color(z, x, y, N_POINTS_PLOT, N_POINTS_PLOT, "Initial condition - PINN")

fig.savefig(os.path.join(output_dir, "initial_condition_pinn_t0.png"))
plt.close(fig)


# --- Generating and Saving Animation Frames ---

frame_files = []
NUMBER_OF_FRAMES = 200

for t_idx in range(NUMBER_OF_FRAMES):
  t_value = TOTAL_TIME * (t_idx / float(NUMBER_OF_FRAMES))
  t_tensor = torch.full_like(x, t_value)
  z = pinn(x, y, t_tensor)
  fig = plot_color(z, x, y, N_POINTS_PLOT, N_POINTS_PLOT, f"PINN Prediction at t={t_value:.2f}")

  frame_filename = os.path.join(animation_dir, f"frame_{t_idx:03d}.png")
  fig.savefig(frame_filename)
  frame_files.append(frame_filename)
  plt.close(fig) # Crucial to close figures in a loop

print(f"Saved {len(frame_files)} animation frames to '{animation_dir}'")


# --- Compiling Frames into a GIF ---

gif_path = os.path.join(output_dir, 'pinn_evolution.gif')
with imageio.get_writer(gif_path, mode='I', fps=15) as writer:
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"Successfully created GIF: '{gif_path}'")