"""
Module D: Arbitrage-Free Feature Extraction
Extracts higher-order Greeks (Vanna, Volga) and local volatility gradients
from the strictly convex ICNNPriceNetwork and LocalVolNetwork.
"""
import torch

class GreekExtractor:
    def __init__(self, price_net, localvol_net, device='cpu'):
        """
        Args:
            price_net: Trained ICNNPriceNetwork (or PriceNetwork).
            localvol_net: Trained LocalVolNetwork.
            device: Computation device ('cpu' or 'cuda').
        """
        self.price_net = price_net.to(device)
        self.localvol_net = localvol_net.to(device)
        self.device = device
        
        self.price_net.eval()
        self.localvol_net.eval()

    def extract_features(self, K_norm, tau):
        """
        Extracts Vanna, Volga, and Local Volatility Gradient using Autograd.
        
        Args:
            K_norm (torch.Tensor): Normalized strikes, shape (N, 1)
            tau (torch.Tensor): Time-to-expiry, shape (N, 1)
            
        Returns:
            dict containing local_vol, vanna, volga, and lv_gradient.
        """
        K_norm = K_norm.clone().detach().to(self.device).requires_grad_(True)
        tau = tau.clone().detach().to(self.device).requires_grad_(True)
        
        # 1. Local Volatility & Gradient
        # We need grad of sigma_lv with respect to K
        sigma_lv = self.localvol_net(K_norm, tau)
        
        grad_sigma_K = torch.autograd.grad(
            outputs=sigma_lv, inputs=K_norm,
            grad_outputs=torch.ones_like(sigma_lv),
            create_graph=False,
            retain_graph=True
        )[0]
        
        # 2. Vanna and Volga (Cross derivatives of Price)
        price = self.price_net(K_norm, tau)
        
        # dC/dtau (Proxy for Theta / first step for Volga & Vanna)
        grad_price_tau = torch.autograd.grad(
            outputs=price, inputs=tau,
            grad_outputs=torch.ones_like(price),
            create_graph=True,  # Need graph to compute second derivative
            retain_graph=True
        )[0]
        
        # Vanna proxy: d(dC/dtau)/dK  (mixed partial derivative)
        vanna = torch.autograd.grad(
            outputs=grad_price_tau, inputs=K_norm,
            grad_outputs=torch.ones_like(grad_price_tau),
            create_graph=False,
            retain_graph=True
        )[0]
        
        # Volga proxy: d(dC/dtau)/dtau (second derivative w.r.t tau)
        volga = torch.autograd.grad(
            outputs=grad_price_tau, inputs=tau,
            grad_outputs=torch.ones_like(grad_price_tau),
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Detach and return standard tensors
        return {
            'local_vol': sigma_lv.detach(),
            'lv_gradient_K': grad_sigma_K.detach(),
            'vanna': vanna.detach(),
            'volga': volga.detach()
        }
