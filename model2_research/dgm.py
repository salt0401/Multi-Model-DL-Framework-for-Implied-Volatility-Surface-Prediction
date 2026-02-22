"""Deep Galerkin Method (DGM) PDE solver for backward Kolmogorov equation.

Implements the DGM network architecture from PDF pages 32-46:
- SLayer: LSTM-like gated layer with Z/G/R/H gates + LayerNorm
- DGMNetwork: Stack of S-layers with residual connections
- DGMLoss: Weighted PDE residual + boundary + terminal condition losses
- DGMSampler: Random sampling of interior/boundary/terminal points
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm


class SLayer(nn.Module):
    """LSTM-like gated layer with Z/G/R/H gates and layer normalization."""

    def __init__(self, input_dim, hidden_dim):
        super(SLayer, self).__init__()

        # Z gate (update gate)
        self.W_z = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # G gate (forget-like)
        self.W_g = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_g = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # R gate (reset gate)
        self.W_r = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # H gate (candidate)
        self.W_h = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, s_prev):
        z = torch.sigmoid(self.W_z(x) + self.U_z(s_prev))
        g = torch.sigmoid(self.W_g(x) + self.U_g(s_prev))
        r = torch.sigmoid(self.W_r(x) + self.U_r(s_prev))
        h = torch.tanh(self.W_h(x) + self.U_h(s_prev * r))

        s_new = (1 - g) * h + z * s_prev
        s_new = self.layer_norm(s_new)
        return s_new


class DGMNetwork(nn.Module):
    """DGM network: stack of S-layers with residual connections.

    Input: (x, t) where x is log-price, t is time
    Output: scalar (option price or transition density)
    """

    def __init__(self, input_dim=2, hidden_dim=64, n_layers=3):
        super(DGMNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # S-layers with residual connections
        self.s_layers = nn.ModuleList([
            SLayer(input_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        """Forward pass.

        Args:
            x: (batch, 1) log-price
            t: (batch, 1) time
        Returns:
            (batch, 1) output
        """
        inputs = torch.cat([x, t], dim=1)  # (batch, 2)
        s = torch.tanh(self.input_proj(inputs))

        for layer in self.s_layers:
            s_new = layer(inputs, s)
            s = s + s_new  # Residual connection

        output = self.output_layer(s)
        return output


class DGMLoss(nn.Module):
    """Weighted PDE residual + boundary + terminal condition losses.

    Solves backward Kolmogorov for GBM:
        du/dt + 0.5*sigma^2*x^2*d2u/dx2 + r*x*du/dx - r*u = 0

    With terminal condition u(x,T) = max(x-K, 0) for calls.
    """

    def __init__(self, sigma=0.2, r=0.0, lambda_pde=1.0, lambda_bc=1.0, lambda_tc=1.0):
        super(DGMLoss, self).__init__()
        self.sigma = sigma
        self.r = r
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_tc = lambda_tc

    def pde_residual(self, model, x, t):
        """Compute PDE residual using autograd."""
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(x, t)

        # First derivatives
        grads = torch.autograd.grad(u.sum(), [x, t], create_graph=True)
        du_dx = grads[0]
        du_dt = grads[1]

        # Second derivative w.r.t. x
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]

        # PDE: du/dt + 0.5*sigma^2*x^2*d2u/dx2 + r*x*du/dx - r*u = 0
        residual = du_dt + 0.5 * self.sigma**2 * x**2 * d2u_dx2 + self.r * x * du_dx - self.r * u

        return residual

    def forward(self, model, x_interior, t_interior, x_boundary, t_boundary, u_boundary,
                x_terminal, u_terminal):
        """Compute total loss.

        Args:
            model: DGMNetwork
            x_interior, t_interior: interior domain points
            x_boundary, t_boundary, u_boundary: boundary condition points
            x_terminal, u_terminal: terminal condition points (t=T)
        """
        # PDE residual loss
        residual = self.pde_residual(model, x_interior, t_interior)
        loss_pde = torch.mean(residual ** 2)

        # Boundary condition loss
        t_T = torch.ones_like(x_terminal) * t_interior.max()
        u_pred_terminal = model(x_terminal, t_T)
        loss_tc = torch.mean((u_pred_terminal - u_terminal) ** 2)

        # Boundary loss (at x boundaries)
        u_pred_boundary = model(x_boundary, t_boundary)
        loss_bc = torch.mean((u_pred_boundary - u_boundary) ** 2)

        total_loss = (self.lambda_pde * loss_pde +
                      self.lambda_bc * loss_bc +
                      self.lambda_tc * loss_tc)

        return total_loss, loss_pde, loss_bc, loss_tc


class DGMSampler:
    """Random sampling of interior/boundary/terminal points in (x, t) domain."""

    def __init__(self, S_min=0.5, S_max=1.5, T_min=0.02, T_max=2.0,
                 n_interior=5000, n_boundary=500, n_terminal=500,
                 strike=1.0, sigma=0.2, r=0.0):
        self.S_min = S_min
        self.S_max = S_max
        self.T_min = T_min
        self.T_max = T_max
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.n_terminal = n_terminal
        self.strike = strike
        self.sigma = sigma
        self.r = r

    def sample(self, device='cpu'):
        """Sample domain points. Returns dict of tensors."""
        dtype = torch.float64

        # Interior points
        x_interior = torch.rand(self.n_interior, 1, dtype=dtype, device=device) * (self.S_max - self.S_min) + self.S_min
        t_interior = torch.rand(self.n_interior, 1, dtype=dtype, device=device) * (self.T_max - self.T_min) + self.T_min

        # Terminal condition: u(x, T) = max(x - K, 0) for calls
        x_terminal = torch.rand(self.n_terminal, 1, dtype=dtype, device=device) * (self.S_max - self.S_min) + self.S_min
        u_terminal = torch.relu(x_terminal - self.strike)

        # Boundary conditions
        # Lower boundary: x = S_min, u ≈ 0 for calls
        n_bc_half = self.n_boundary // 2
        t_bc_low = torch.rand(n_bc_half, 1, dtype=dtype, device=device) * (self.T_max - self.T_min) + self.T_min
        x_bc_low = torch.full((n_bc_half, 1), self.S_min, dtype=dtype, device=device)
        u_bc_low = torch.relu(x_bc_low - self.strike)

        # Upper boundary: x = S_max, u ≈ x - K*exp(-r*t) for calls deep ITM
        t_bc_high = torch.rand(n_bc_half, 1, dtype=dtype, device=device) * (self.T_max - self.T_min) + self.T_min
        x_bc_high = torch.full((n_bc_half, 1), self.S_max, dtype=dtype, device=device)
        u_bc_high = x_bc_high - self.strike * torch.exp(-self.r * t_bc_high)

        x_boundary = torch.cat([x_bc_low, x_bc_high], dim=0)
        t_boundary = torch.cat([t_bc_low, t_bc_high], dim=0)
        u_boundary = torch.cat([u_bc_low, u_bc_high], dim=0)

        return {
            'x_interior': x_interior,
            't_interior': t_interior,
            'x_terminal': x_terminal,
            'u_terminal': u_terminal,
            'x_boundary': x_boundary,
            't_boundary': t_boundary,
            'u_boundary': u_boundary,
        }


def price_european_option(model, S, K, T, r=0.0, device='cpu', option_type='call'):
    """Use trained DGM to price European options.

    Args:
        model: trained DGMNetwork
        S: spot price (scalar or array)
        K: strike price
        T: time to maturity
        r: risk-free rate
        device: torch device
        option_type: 'call' or 'put'
    Returns:
        option price(s)
    """
    model.eval()
    dtype = torch.float64

    if np.isscalar(S):
        S = np.array([S])
    x = torch.tensor(S / K, dtype=dtype, device=device).reshape(-1, 1)
    t = torch.full_like(x, T)

    with torch.no_grad():
        price = model(x, t).cpu().numpy().flatten() * K

    if option_type == 'put':
        # Put-call parity: P = C - S + K*exp(-r*T)
        price = price - S + K * np.exp(-r * T)

    return price


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price for validation."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
