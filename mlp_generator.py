"""
CALM MLPGenerator — noise-conditioned generative head.

Standalone PyTorch implementation with no HuggingFace dependencies.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPGeneratorConfig:
    hidden_size: int = 768
    latent_size: int = 128
    noise_size: int = 64
    num_mlp_layers: int = 4


class MLPBlock(nn.Module):
    """Residual block that refines noise embedding x conditioned on hidden state y."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(2 * channels, channels, bias=True)
        self.linear2 = nn.Linear(channels, channels, bias=True)
        self.linear3 = nn.Linear(channels, 2 * channels, bias=True)
        self.down_proj = nn.Linear(channels, channels, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.norm(x), y], dim=-1)
        h = self.silu(self.linear1(h))
        h = self.silu(self.linear2(h))
        h = self.linear3(h)

        gate_proj, up_proj = h.chunk(2, dim=-1)
        gate_proj = self.silu(gate_proj)
        step = self.down_proj(gate_proj * up_proj)
        return x + step


class FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_channels)
        self.linear1 = nn.Linear(model_channels, model_channels, bias=True)
        self.linear2 = nn.Linear(model_channels, out_channels, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.silu(self.linear1(x))
        return self.linear2(x)


class MLPGenerator(nn.Module):
    def __init__(self, config: MLPGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        hs = config.hidden_size

        self.noise_embd = nn.Linear(config.noise_size, hs, bias=True)
        self.hidden_embd = nn.Linear(hs, hs, bias=True)
        self.norm_hidden = nn.LayerNorm(hs)
        self.norm_noise = nn.LayerNorm(hs)
        self.mlp_blocks = nn.ModuleList(
            [MLPBlock(hs) for _ in range(config.num_mlp_layers)]
        )
        self.final_layer = FinalLayer(hs, config.latent_size)

    def initialize_weights(self) -> None:
        """Zero-initialize the last linear layer of final_layer for stable training."""
        nn.init.zeros_(self.final_layer.linear2.weight)
        nn.init.zeros_(self.final_layer.linear2.bias)

    def sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        noise_shape = (*hidden_states.shape[:-1], self.config.noise_size)
        noise = torch.rand(noise_shape, device=hidden_states.device, dtype=hidden_states.dtype) - 0.5

        noise_embds = self.norm_noise(self.noise_embd(noise))
        hidden_states = self.norm_hidden(self.hidden_embd(hidden_states))

        for block in self.mlp_blocks:
            noise_embds = block(noise_embds, hidden_states)

        return self.final_layer(noise_embds)


if __name__ == "__main__":
    cfg = MLPGeneratorConfig()
    model = MLPGenerator(cfg)
    model.initialize_weights()

    print(f"Config: {cfg}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    hidden_states = torch.randn(2, 10, cfg.hidden_size)
    out = model.sample(hidden_states)
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, 10, cfg.latent_size), f"Unexpected shape {out.shape}"

    # Verify gradient flow
    loss = out.sum()
    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    assert len(grad_norms) > 0, "No gradients computed"
    print(f"Gradient check passed — {len(grad_norms)} params have gradients")
    print("All checks passed ✓")
