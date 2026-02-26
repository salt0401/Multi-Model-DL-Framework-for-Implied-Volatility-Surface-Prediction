"""Dupire PINN: Physics-Informed Neural Network for local volatility extraction.

V1 prototype — soft-constraint PINN with standard MLP architecture.
Implements dual-network design:
  - PriceNetwork: corrected call price C(K, T)
  - LocalVolNetwork: local variance σ²_LV(K, T)
  - DupirePINNLoss: 5-term loss with Dupire PDE residual
  - DupireSampler: collocation point sampling in (K, τ) domain

The two networks are jointly trained to satisfy the Dupire PDE:
    ∂C/∂T = ½ σ²_LV K² ∂²C/∂K²

References:
    - Wang & Privault (2022/2025), arXiv:2201.07880
    - WamOL (ICAIF 2024), arXiv:2411.02375
    - Bae, Kang & Lee (2024), Computational Economics 64:3143
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm


# ── Price Network ─────────────────────────────────────────────────────

class PriceNetwork(nn.Module):
    """Predicts corrected European call price C(K, T) ≥ 0.

    Input: (K_norm, tau) where K_norm is normalized strike price
    Output: call price (always non-negative via softplus)
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        super(PriceNetwork, self).__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)

        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for _ in range(n_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            # Skip connection from input dimension
            self.skip_projs.append(nn.Linear(2, hidden_dim, bias=False))

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus()

    def forward(self, K, tau):
        """Forward pass.

        Args:
            K: (batch, 1) normalized strike price
            tau: (batch, 1) time-to-expiry

        Returns:
            (batch, 1) call price, guaranteed ≥ 0
        """
        x_input = torch.cat([K, tau], dim=1)  # (batch, 2)
        h = self.activation(self.layer_norm_input(self.input_proj(x_input)))

        for layer, ln, skip in zip(self.hidden_layers, self.layer_norms, self.skip_projs):
            h_new = self.activation(ln(layer(h) + skip(x_input)))
            h = h + h_new  # Residual connection

        return self.activation(self.output_layer(h))  # softplus ensures C ≥ 0


class ICNNPriceNetwork(nn.Module):
    """Predicts corrected European call price C(K, T) ≥ 0 with hard convexity in K.

    Input: (K_norm, tau) where K_norm is normalized strike price
    Output: call price (always non-negative via softplus)

    Guarantees:
        ∂²C/∂K² ≥ 0 exactly everywhere, ensuring no butterfly arbitrage.
        This is achieved by restricting weights connecting hidden layers,
        and weights connecting K to hidden layers, to be non-negative.
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        super(ICNNPriceNetwork, self).__init__()
        self.n_layers = n_layers

        # W_x_K: weights from K to hidden layers (must be non-negative)
        # W_x_tau: weights from tau to hidden layers (unconstrained)
        self.W_x_K = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, 1) / np.sqrt(1)) for _ in range(n_layers + 1)])
        self.W_x_tau = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, 1) / np.sqrt(1)) for _ in range(n_layers + 1)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(n_layers + 1)])

        # W_z: weights between hidden layers (must be non-negative)
        self.W_z = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)) for _ in range(n_layers)])

        # Output layer weights
        self.W_out_z = nn.Parameter(torch.randn(1, hidden_dim) / np.sqrt(hidden_dim))
        self.W_out_x_K = nn.Parameter(torch.randn(1, 1))
        self.W_out_x_tau = nn.Parameter(torch.randn(1, 1))
        self.b_out = nn.Parameter(torch.zeros(1))

        self.activation = nn.Softplus()

    def forward(self, K, tau):
        """Forward pass.

        Args:
            K: (batch, 1) normalized strike price
            tau: (batch, 1) time-to-expiry

        Returns:
            (batch, 1) call price, guaranteed ≥ 0, convex in K
        """
        # Ensure non-negative weights via softplus during forward pass
        # This acts as weight projection without breaking autograd
        
        # Layer 0
        w_x_K_0 = torch.nn.functional.softplus(self.W_x_K[0])
        h = self.activation(torch.matmul(K, w_x_K_0.t()) + torch.matmul(tau, self.W_x_tau[0].t()) + self.b[0])

        # Hidden layers
        for i in range(self.n_layers):
            w_z = torch.nn.functional.softplus(self.W_z[i])
            w_x_K = torch.nn.functional.softplus(self.W_x_K[i+1])
            z_proj = torch.matmul(h, w_z.t())
            x_proj = torch.matmul(K, w_x_K.t()) + torch.matmul(tau, self.W_x_tau[i+1].t()) + self.b[i+1]
            h = self.activation(z_proj + x_proj)
            
        # Output layer
        w_out_z = torch.nn.functional.softplus(self.W_out_z)
        w_out_x_K = torch.nn.functional.softplus(self.W_out_x_K)
        out = torch.matmul(h, w_out_z.t()) + torch.matmul(K, w_out_x_K.t()) + torch.matmul(tau, self.W_out_x_tau.t()) + self.b_out
        
        return self.activation(out)


# ── Local Volatility Network ─────────────────────────────────────────

class LocalVolNetwork(nn.Module):
    """Predicts local variance σ²_LV(K, T) > 0.

    Input: (K_norm, tau)
    Output: local variance (strictly positive via softplus + epsilon)
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        super(LocalVolNetwork, self).__init__()
        self.eps = 1e-8

        self.input_proj = nn.Linear(2, hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)

        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for _ in range(n_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.skip_projs.append(nn.Linear(2, hidden_dim, bias=False))

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus()

    def forward(self, K, tau):
        """Forward pass.

        Args:
            K: (batch, 1) normalized strike price
            tau: (batch, 1) time-to-expiry

        Returns:
            (batch, 1) local variance σ²_LV, guaranteed > 0
        """
        x_input = torch.cat([K, tau], dim=1)  # (batch, 2)
        h = self.activation(self.layer_norm_input(self.input_proj(x_input)))

        for layer, ln, skip in zip(self.hidden_layers, self.layer_norms, self.skip_projs):
            h_new = self.activation(ln(layer(h) + skip(x_input)))
            h = h + h_new

        return self.activation(self.output_layer(h)) + self.eps  # strictly > 0


# ── Dupire PINN Loss ─────────────────────────────────────────────────

class DupirePINNLoss(nn.Module):
    """5-term weighted loss for Dupire PDE-constrained local vol extraction.

    L = λ_fit · L_fit + λ_pde · L_dupire + λ_cal · L_calendar
        + λ_but · L_butterfly + λ_smooth · L_smooth

    The PDE residual is computed via autograd:
        Dupire PDE: ∂C/∂T = ½ σ²_LV K² ∂²C/∂K²
    """

    def __init__(self, lambda_fit=1.0, lambda_pde=10.0, lambda_cal=10.0,
                 lambda_but=10.0, lambda_smooth=1.0):
        super(DupirePINNLoss, self).__init__()
        self.lambda_fit = lambda_fit
        self.lambda_pde = lambda_pde
        self.lambda_cal = lambda_cal
        self.lambda_but = lambda_but
        self.lambda_smooth = lambda_smooth

    def dupire_pde_residual(self, price_net, localvol_net, K, tau):
        """Compute Dupire PDE residual using autograd.

        Dupire PDE: ∂C/∂T - ½ σ²_LV K² ∂²C/∂K² = 0

        Args:
            price_net: PriceNetwork
            localvol_net: LocalVolNetwork
            K: (batch, 1) normalized strike, requires_grad=True
            tau: (batch, 1) time-to-expiry, requires_grad=True

        Returns:
            pde_residual: (batch, 1)
            dC_dT: (batch, 1) for calendar constraint
            d2C_dK2: (batch, 1) for butterfly constraint
            sigma2_LV: (batch, 1) local variance
        """
        C = price_net(K, tau)

        # ∂C/∂T
        dC_dT = torch.autograd.grad(
            outputs=C, inputs=tau,
            grad_outputs=torch.ones_like(C),
            create_graph=True, retain_graph=True
        )[0]

        # ∂C/∂K
        dC_dK = torch.autograd.grad(
            outputs=C, inputs=K,
            grad_outputs=torch.ones_like(C),
            create_graph=True, retain_graph=True
        )[0]

        # ∂²C/∂K²
        d2C_dK2 = torch.autograd.grad(
            outputs=dC_dK, inputs=K,
            grad_outputs=torch.ones_like(dC_dK),
            create_graph=True, retain_graph=True
        )[0]

        # σ²_LV
        sigma2_LV = localvol_net(K, tau)

        # Dupire PDE residual: ∂C/∂T - ½ σ²_LV K² ∂²C/∂K²
        pde_residual = dC_dT - 0.5 * sigma2_LV * K ** 2 * d2C_dK2

        return pde_residual, dC_dT, d2C_dK2, sigma2_LV

    def forward(self, price_net, localvol_net, K, tau, C_target):
        """Compute total loss.

        Args:
            price_net: PriceNetwork
            localvol_net: LocalVolNetwork
            K: (batch, 1) normalized strike, requires_grad=True
            tau: (batch, 1) time-to-expiry, requires_grad=True
            C_target: (batch, 1) target call prices (from Model 1 or BS)

        Returns:
            (total_loss, loss_fit, loss_pde, loss_cal, loss_but, loss_smooth)
        """
        # PDE residual and derivatives
        pde_residual, dC_dT, d2C_dK2, sigma2_LV = \
            self.dupire_pde_residual(price_net, localvol_net, K, tau)

        # (1) Fit loss: MSE between predicted and target prices
        C_pred = price_net(K, tau)
        loss_fit = torch.mean((C_pred - C_target) ** 2)

        # (2) Dupire PDE residual
        loss_pde = torch.mean(pde_residual ** 2)

        # (3) Calendar arbitrage: ∂C/∂T ≥ 0 (longer expiry = more value)
        loss_cal = torch.mean(torch.relu(-dC_dT))

        # (4) Butterfly arbitrage: ∂²C/∂K² ≥ 0 (price convex in K)
        loss_but = torch.mean(torch.relu(-d2C_dK2))

        # (5) Local vol smoothness: penalize large gradients
        # Compute ∂σ²_LV/∂K and ∂σ²_LV/∂T
        dsig_dK = torch.autograd.grad(
            outputs=sigma2_LV, inputs=K,
            grad_outputs=torch.ones_like(sigma2_LV),
            create_graph=True, retain_graph=True
        )[0]
        dsig_dT = torch.autograd.grad(
            outputs=sigma2_LV, inputs=tau,
            grad_outputs=torch.ones_like(sigma2_LV),
            create_graph=True, retain_graph=True
        )[0]
        loss_smooth = torch.mean(dsig_dK ** 2 + dsig_dT ** 2)

        # Total
        total_loss = (
            self.lambda_fit * loss_fit
            + self.lambda_pde * loss_pde
            + self.lambda_cal * loss_cal
            + self.lambda_but * loss_but
            + self.lambda_smooth * loss_smooth
        )

        return total_loss, loss_fit, loss_pde, loss_cal, loss_but, loss_smooth


# ── Dupire Sampler ────────────────────────────────────────────────────

class DupireSampler:
    """Random sampling of collocation points in (K, tau) domain.

    Generates interior points for PDE constraint and boundary points
    for boundary conditions (tau=0 payoff, K extremes).

    Two modes:
      - Synthetic BS mode (default): uses Black-Scholes formula with fixed sigma
        for pipeline validation.
      - Model 1 mode: queries a pre-trained MultiModel to get total variance
        predictions, then converts them to call prices. This is the production
        pipeline for real training.
    """

    def __init__(self, K_min=0.5, K_max=1.5, tau_min=0.02, tau_max=2.0,
                 n_interior=5000, n_boundary=500, strike=1.0, sigma_bs=0.2,
                 base_model=None, yATM=None):
        """Initialize sampler.

        Args:
            K_min: minimum normalized strike
            K_max: maximum normalized strike
            tau_min: minimum time-to-expiry
            tau_max: maximum time-to-expiry
            n_interior: number of interior collocation points
            n_boundary: number of boundary points
            strike: spot price (normalized, default 1.0)
            sigma_bs: BS volatility for synthetic mode (ignored in Model 1 mode)
            base_model: (optional) pre-trained Model 1 MultiModel instance.
                        If provided, switches to Model 1 mode.
            yATM: (optional) ATM total variance scalar or tensor.
                  Required when base_model is provided.
        """
        self.K_min = K_min
        self.K_max = K_max
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.strike = strike
        self.sigma_bs = sigma_bs
        self.base_model = base_model
        self.yATM = yATM

        if base_model is not None and yATM is None:
            raise ValueError(
                "yATM must be provided when using Model 1 mode. "
                "Pass a scalar (e.g. mean ATM total variance from the dataset)."
            )

    def _query_model1(self, K, tau, device):
        """Query Model 1 for call prices at given (K, tau) points.

        Pipeline: K → logm → MultiModel(tau, logm, yATM) → tv_pred → C_target

        Note: Model 1's SmileModel internally uses autograd.grad(create_graph=True)
        to compute first/second derivatives, so inputs MUST have requires_grad=True
        and we cannot wrap the call in torch.no_grad().

        Args:
            K: (n, 1) normalized strike prices (detached)
            tau: (n, 1) time-to-expiry (detached)
            device: torch device

        Returns:
            C_target: (n, 1) call prices derived from Model 1's total variance
        """
        # Convert K to log-moneyness: logm = ln(K / S), S = strike = 1.0
        logm = torch.log(K / self.strike)

        # Prepare yATM broadcast to batch size
        if isinstance(self.yATM, (int, float)):
            yATM_batch = torch.full_like(tau, self.yATM)
        else:
            yATM_batch = self.yATM.expand_as(tau)

        # SmileModel.forward uses autograd.grad(create_graph=True) internally,
        # so inputs must have requires_grad=True. We enable grad, query, then detach.
        tau_q = tau.clone().requires_grad_(True)
        logm_q = logm.clone().requires_grad_(True)

        tv_pred, _, _, _ = self.base_model(tau_q, logm_q, yATM_batch)
        tv_pred = tv_pred.detach()  # Sever Model 1's computation graph

        # Convert total variance → call price via Black-Scholes
        C_target = _total_variance_to_call_price_tensor(
            tv_pred, K, self.strike, tau
        )
        return C_target

    def sample(self, device='cpu'):
        """Sample domain points. Returns dict of tensors.

        Interior points: random (K, tau) in domain, with requires_grad=True
        Boundary points at tau → 0: C(K, 0) = max(S - K, 0)
        Target prices: from Model 1 (production) or Black-Scholes (validation)

        Returns:
            dict with keys:
                K_interior, tau_interior: (n_interior, 1) with requires_grad
                K_boundary, tau_boundary, C_boundary: (n_boundary, 1)
                C_target: (n_interior, 1) target call prices for fit loss
        """
        # Interior points: (K, tau) in domain
        K_int = (torch.rand(self.n_interior, 1) *
                 (self.K_max - self.K_min) + self.K_min).to(device)
        tau_int = (torch.rand(self.n_interior, 1) *
                   (self.tau_max - self.tau_min) + self.tau_min).to(device)
        K_int.requires_grad_(True)
        tau_int.requires_grad_(True)

        # Compute target prices
        if self.base_model is not None:
            # ── Model 1 mode: query pre-trained MultiModel ──
            C_target = self._query_model1(
                K_int.detach(), tau_int.detach(), device
            ).to(device)
        else:
            # ── Synthetic BS mode: for pipeline validation only ──
            C_target = _bs_call_price_tensor(
                K_int.detach(), self.strike, tau_int.detach(), self.sigma_bs
            ).to(device)

        # Boundary points at tau → 0: payoff = max(S - K, 0)
        K_bnd = (torch.rand(self.n_boundary, 1) *
                 (self.K_max - self.K_min) + self.K_min).to(device)
        tau_bnd = torch.full((self.n_boundary, 1), self.tau_min,
                             device=device)
        C_bnd = torch.relu(self.strike - K_bnd)  # Intrinsic value payoff

        return {
            'K_interior': K_int,
            'tau_interior': tau_int,
            'C_target': C_target,
            'K_boundary': K_bnd,
            'tau_boundary': tau_bnd,
            'C_boundary': C_bnd,
        }


# ── Utility Functions ─────────────────────────────────────────────────

def _bs_call_price_tensor(K, S, tau, sigma, r=0.0):
    """Black-Scholes call price (tensor version for sampler targets).

    Args:
        K: (batch, 1) strike prices (tensor)
        S: scalar spot price
        tau: (batch, 1) time to expiry (tensor)
        sigma: scalar implied volatility
        r: risk-free rate

    Returns:
        (batch, 1) call prices
    """
    # Convert to numpy for norm.cdf, then back
    K_np = K.detach().cpu().numpy()
    tau_np = tau.detach().cpu().numpy()
    tau_np = np.maximum(tau_np, 1e-10)  # avoid division by zero

    d1 = (np.log(S / K_np) + (r + 0.5 * sigma ** 2) * tau_np) / (sigma * np.sqrt(tau_np))
    d2 = d1 - sigma * np.sqrt(tau_np)

    price = S * norm.cdf(d1) - K_np * np.exp(-r * tau_np) * norm.cdf(d2)
    return torch.tensor(price, dtype=torch.float64)


def _total_variance_to_call_price_tensor(w, K, S, tau, r=0.0):
    """Convert Model 1's total variance to call prices via BS formula.

    This is the tensor-compatible bridge between Model 1 output (total variance)
    and Model 2 input (call prices). Uses numpy internally for norm.cdf.

    Args:
        w: (batch, 1) total variance = IV² × τ (tensor)
        K: (batch, 1) strike prices (tensor)
        S: scalar spot price
        tau: (batch, 1) time to expiry (tensor)
        r: risk-free rate

    Returns:
        (batch, 1) call prices
    """
    w_np = w.detach().cpu().numpy()
    K_np = K.detach().cpu().numpy()
    tau_np = tau.detach().cpu().numpy()
    tau_np = np.maximum(tau_np, 1e-10)

    # total variance → implied volatility
    iv = np.sqrt(np.maximum(w_np, 0) / tau_np)
    iv = np.maximum(iv, 1e-10)  # avoid division by zero

    d1 = (np.log(S / K_np) + (r + 0.5 * iv ** 2) * tau_np) / (iv * np.sqrt(tau_np))
    d2 = d1 - iv * np.sqrt(tau_np)

    price = S * norm.cdf(d1) - K_np * np.exp(-r * tau_np) * norm.cdf(d2)
    return torch.tensor(price, dtype=torch.float64)


def total_variance_to_call_price(w, K, S, tau, r=0.0):
    """Convert Model 1's total variance to call prices via BS formula.

    Args:
        w: total variance (IV² × τ)
        K: strike price
        S: spot price
        tau: time to expiry
        r: risk-free rate

    Returns:
        call prices
    """
    iv = np.sqrt(np.maximum(w, 0) / np.maximum(tau, 1e-10))
    return bs_call_price(S, K, tau, r, iv)


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price for validation (numpy version).

    Args:
        S: spot price (scalar or array)
        K: strike price (scalar or array)
        T: time to maturity
        r: risk-free rate
        sigma: implied volatility

    Returns:
        call price(s)
    """
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
