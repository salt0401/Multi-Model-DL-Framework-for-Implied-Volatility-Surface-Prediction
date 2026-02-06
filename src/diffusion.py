"""Conditional DDPM for IV Surface Forecasting.

Denoising Diffusion Probabilistic Model conditioned on current surface +
market features to generate next-day IV surface.

The IV surface is represented as a 1D vector on a fixed (tau, logm) grid.
A 1D U-Net denoises noisy surfaces conditioned on market state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal encoding for diffusion timestep t."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        Args:
            t: (batch,) integer timesteps
        Returns:
            (batch, embed_dim)
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=t.dtype, device=t.device) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta."""

    def __init__(self, condition_dim, channels):
        super().__init__()
        self.proj = nn.Linear(condition_dim, channels * 2)

    def forward(self, x, condition):
        """
        Args:
            x: (batch, channels, length)
            condition: (batch, condition_dim)
        """
        gamma_beta = self.proj(condition)  # (batch, channels*2)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # (batch, channels, 1)
        beta = beta.unsqueeze(-1)
        return gamma * x + beta


class ResBlock1D(nn.Module):
    """1D residual block with GroupNorm + SiLU + Conv1d + FiLM conditioning."""

    def __init__(self, in_channels, out_channels, time_dim, cond_dim, n_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(n_groups, in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(n_groups, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.film = FiLMLayer(cond_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, t_emb, cond):
        """
        Args:
            x: (batch, in_channels, length)
            t_emb: (batch, time_dim)
            cond: (batch, cond_dim)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb).unsqueeze(-1)

        # FiLM conditioning
        h = self.film(h, cond)

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


class UNet1D(nn.Module):
    """1D U-Net denoising backbone for IV surface diffusion.

    Encoder: 3 downsampling stages (Conv1d stride 2)
    Bottleneck: 2 ResBlocks
    Decoder: 3 upsampling stages (ConvTranspose1d)
    Skip connections between encoder and decoder.
    """

    def __init__(self, surface_dim, channels=(64, 128, 256), condition_dim=4, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition encoder: current surface + market features -> cond embedding
        self.cond_encoder = nn.Sequential(
            nn.Linear(surface_dim + condition_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_ch = channels[0]
        for ch in channels:
            self.encoder_blocks.append(ResBlock1D(in_ch, ch, time_dim, time_dim))
            self.downsample.append(nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1))
            in_ch = ch

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock1D(channels[-1], channels[-1], time_dim, time_dim),
            ResBlock1D(channels[-1], channels[-1], time_dim, time_dim),
        ])

        # Decoder
        self.upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            up_in = channels[-1] if i == 0 else channels[len(channels) - i]
            self.upsample.append(nn.ConvTranspose1d(up_in, ch, kernel_size=4, stride=2, padding=1))
            # Skip connection doubles channels
            self.decoder_blocks.append(ResBlock1D(ch * 2, ch, time_dim, time_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], 1, kernel_size=3, padding=1),
        )

    def forward(self, x_noisy, t, current_surface, market_features):
        """
        Args:
            x_noisy:         (batch, surface_dim) — noisy target surface
            t:               (batch,) — diffusion timestep
            current_surface: (batch, surface_dim) — today's IV surface
            market_features: (batch, condition_dim) — [VIX, VIX_change, return, rvol]

        Returns:
            noise_pred: (batch, surface_dim) — predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_dim)

        # Condition embedding
        cond_input = torch.cat([current_surface, market_features], dim=-1)
        cond = self.cond_encoder(cond_input)  # (batch, time_dim)

        # Reshape to (batch, 1, surface_dim) for 1D conv
        h = x_noisy.unsqueeze(1)
        h = self.input_proj(h)  # (batch, channels[0], surface_dim)

        # Encoder with skip connections
        skips = []
        for enc_block, down in zip(self.encoder_blocks, self.downsample):
            h = enc_block(h, t_emb, cond)
            skips.append(h)
            h = down(h)

        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb, cond)

        # Decoder
        for up, dec_block, skip in zip(self.upsample, self.decoder_blocks, reversed(skips)):
            h = up(h)
            # Handle size mismatch from stride-2 downsampling
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = dec_block(h, t_emb, cond)

        # Output
        h = self.output_proj(h)  # (batch, 1, surface_dim)
        return h.squeeze(1)  # (batch, surface_dim)


class CosineNoiseSchedule:
    """Cosine beta schedule (Nichol & Dhariwal 2021).

    Better than linear for structured data like IV surfaces.
    """

    def __init__(self, T=1000, s=0.008):
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]

        # Clip betas to prevent singularities
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        beta = torch.clamp(beta, min=1e-5, max=0.999)

        self.beta = beta
        self.alpha = 1 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def to(self, device):
        """Move all schedule tensors to device."""
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self


class DiffusionTrainer:
    """Training logic for the conditional DDPM."""

    def __init__(self, model, schedule, optimizer, device):
        self.model = model
        self.schedule = schedule
        self.optimizer = optimizer
        self.device = device

    def train_step(self, x_target, current_surface, market_features):
        """One training step.

        Args:
            x_target:        (batch, surface_dim) — tomorrow's IV surface
            current_surface: (batch, surface_dim) — today's IV surface
            market_features: (batch, condition_dim)

        Returns:
            loss value (float)
        """
        self.model.train()
        batch_size = x_target.shape[0]

        # Sample random timestep
        t = torch.randint(0, self.schedule.T, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_target)

        # Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise
        sqrt_ab = self.schedule.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.schedule.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_noisy = sqrt_ab * x_target + sqrt_omab * noise

        # Predict noise
        noise_pred = self.model(x_noisy, t.float(), current_surface, market_features)

        # MSE loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


class DiffusionSampler:
    """DDPM reverse process sampling."""

    def __init__(self, model, schedule, device):
        self.model = model
        self.schedule = schedule
        self.device = device

    @torch.no_grad()
    def sample(self, current_surface, market_features, n_samples=1):
        """Generate next-day IV surfaces via reverse diffusion.

        Args:
            current_surface: (1, surface_dim) or (n_samples, surface_dim)
            market_features: (1, condition_dim) or (n_samples, condition_dim)
            n_samples: number of samples to generate

        Returns:
            (n_samples, surface_dim)
        """
        self.model.eval()
        surface_dim = current_surface.shape[-1]

        if current_surface.shape[0] == 1 and n_samples > 1:
            current_surface = current_surface.expand(n_samples, -1)
            market_features = market_features.expand(n_samples, -1)

        # Start from pure noise
        x = torch.randn(n_samples, surface_dim, dtype=current_surface.dtype, device=self.device)

        for t_idx in reversed(range(self.schedule.T)):
            t = torch.full((n_samples,), t_idx, dtype=current_surface.dtype, device=self.device)

            noise_pred = self.model(x, t, current_surface, market_features)

            alpha = self.schedule.alpha[t_idx]
            alpha_bar = self.schedule.alpha_bar[t_idx]
            beta = self.schedule.beta[t_idx]

            # DDPM update
            mean = (1 / torch.sqrt(alpha)) * (
                x - beta / torch.sqrt(1 - alpha_bar) * noise_pred
            )

            if t_idx > 0:
                sigma = torch.sqrt(beta)
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x
