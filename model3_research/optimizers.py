"""Custom optimizers for Model 3 regularization experiments.

Implements:
- CautiousAdamW: AdamW with Cautious Weight Decay (CWD, ICLR 2026)
- AdamCPR: Adam with Constrained Parameter Regularization (NeurIPS 2024)

Both are drop-in replacements for torch.optim.Adam / AdamW.
"""
import math
import torch
from torch.optim import Optimizer


class CautiousAdamW(Optimizer):
    """AdamW with Cautious Weight Decay (CWD).

    Only applies weight decay when the Adam update direction aligns with
    the parameter's sign (per-coordinate), preventing decay from fighting
    the optimizer when it's pushing parameters away from zero.

    Reference:
        Cautious Weight Decay, ICLR 2026, arXiv:2510.12402

    Standard AdamW applies decay unconditionally:
        x_{t+1} = x_t - lr * (u_t + λ * x_t)

    CWD adds a per-parameter mask:
        x_{t+1} = x_t - lr * (u_t + λ * 𝕀(u_t ⊙ x_t >= 0) ⊙ x_t)

    When the optimizer wants to push a parameter away from zero (u*x < 0),
    weight decay is suppressed for that parameter.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, cautious=True):
        """
        Args:
            params: iterable of parameters or param groups
            lr: learning rate
            betas: (beta1, beta2) for Adam moment estimates
            eps: numerical stability term
            weight_decay: decoupled weight decay coefficient
            cautious: if True, use CWD mask; if False, behave as standard AdamW
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, cautious=cautious)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            cautious = group['cautious']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('CautiousAdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                step_size = lr / bc1

                # Compute Adam update direction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                update = exp_avg / denom

                # === Weight Decay (with optional CWD mask) ===
                if wd > 0:
                    if cautious:
                        # CWD: only decay when update and parameter agree
                        mask = (update * p.data >= 0).to(p.data.dtype)
                        p.data.add_(p.data * mask, alpha=-lr * wd)
                    else:
                        # Standard decoupled weight decay (AdamW)
                        p.data.add_(p.data, alpha=-lr * wd)

                # Apply gradient step
                p.data.add_(update, alpha=-step_size)

        return loss


class AdamCPR(Optimizer):
    """Adam with Constrained Parameter Regularization (CPR).

    Instead of applying a fixed weight decay λ to all parameters,
    CPR enforces an upper bound κ on the L2-norm of each parameter group.
    Uses the augmented Lagrangian method to adaptively adjust the
    regularization strength per parameter group.

    Reference:
        Franke et al., "Improving Deep Learning Optimization through
        Constrained Parameter Regularization", NeurIPS 2024

    Key advantages over standard weight decay:
    - No need to tune λ — only set κ (upper bound), with auto-init
    - Each parameter group gets independent, adaptive regularization
    - Outperforms AdamW in GPT2 and DeiT training
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 kappa_init_method='uniform', kappa_init_param=1.0,
                 mu=0.01, lam_lr=0.1):
        """
        Args:
            params: iterable of parameters or param groups
            lr: learning rate
            betas: (beta1, beta2) for Adam moments
            eps: numerical stability
            kappa_init_method: 'uniform' (same κ for all) or 'warm_start'
                (set κ = current norm * kappa_init_param after first step)
            kappa_init_param: multiplier for κ initialization
            mu: augmented Lagrangian penalty coefficient
            lam_lr: learning rate for Lagrangian multiplier updates
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        kappa_init_method=kappa_init_method,
                        kappa_init_param=kappa_init_param,
                        mu=mu, lam_lr=lam_lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step with CPR constraint."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            mu = group['mu']
            lam_lr = group['lam_lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamCPR does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['lam'] = 0.0  # Lagrangian multiplier

                    # Initialize kappa (upper bound on norm)
                    if group['kappa_init_method'] == 'warm_start':
                        # Set kappa based on current parameter norm
                        state['kappa'] = float(p.data.norm()) * group['kappa_init_param']
                    else:
                        state['kappa'] = group['kappa_init_param']

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                step_size = lr / bc1

                # Compute Adam update direction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                update = exp_avg / denom

                # Apply gradient step (standard Adam)
                p.data.add_(update, alpha=-step_size)

                # === CPR: Augmented Lagrangian constraint enforcement ===
                param_norm = float(p.data.norm())
                kappa = state['kappa']
                violation = param_norm - kappa

                if violation > 0:
                    # Norm exceeds bound: apply penalty + update multiplier
                    # Penalty: project towards feasible region
                    penalty_strength = state['lam'] + mu * violation
                    # Scale parameter towards norm = kappa
                    scale = kappa / (param_norm + eps)
                    # Soft projection: blend current with projected
                    blend = min(lr * penalty_strength, 0.5)  # cap to prevent collapse
                    p.data.mul_(1.0 - blend + blend * scale)
                    # Update Lagrangian multiplier
                    state['lam'] = max(0.0, state['lam'] + lam_lr * violation)
                else:
                    # Within bound: relax multiplier towards zero
                    state['lam'] = max(0.0, state['lam'] + lam_lr * violation)

        return loss
