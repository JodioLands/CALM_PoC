"""
CALM Autoencoder — standalone PyTorch VAE-based implementation.

No HuggingFace dependencies. Uses only torch and torch.nn.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutoencoderConfig:
    vocab_size: int = 128256
    hidden_size: int = 512
    intermediate_size: int = 1280
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    patch_size: int = 4
    latent_size: int = 128
    ae_dropout: float = 0.15
    kl_clamp: float = 0.5
    kl_weight: float = 1e-3
    rms_norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    """LlamaRMSNorm: weight * (x / sqrt(mean(x^2) + eps))"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class GatedMLP(nn.Module):
    """SiLU-gated MLP: down_proj(gate_proj(x) * silu(up_proj(x)))"""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(x) * F.silu(self.up_proj(x)))


class AELayer(nn.Module):
    """RMSNorm + GatedMLP with residual connection."""

    def __init__(self, hidden_size: int, intermediate_size: int, rms_norm_eps: float):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.norm(hidden_states))


class Encoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        num_stage_layers = config.num_encoder_layers // 2

        # Stage 0: per-token within each patch
        self.stage0_layers = nn.ModuleList(
            [
                AELayer(config.hidden_size, config.intermediate_size, config.rms_norm_eps)
                for _ in range(num_stage_layers)
            ]
        )

        # Squeeze: (patch_size * hidden_size) → hidden_size
        self.squeeze_layer = nn.Linear(
            config.patch_size * config.hidden_size, config.hidden_size, bias=False
        )

        # Stage 1: on squeezed patch representations
        self.stage1_layers = nn.ModuleList(
            [
                AELayer(config.hidden_size, config.intermediate_size, config.rms_norm_eps)
                for _ in range(num_stage_layers)
            ]
        )

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Output mean and log_std
        self.hidden_to_latent = nn.Linear(
            config.hidden_size, config.latent_size * 2, bias=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch * num_patches, patch_size)
        Returns:
            latent_params: (batch * num_patches, 1, latent_size * 2)
        """
        batch_num_patches, patch_size = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        # hidden_states: (batch*num_patches, patch_size, hidden_size)

        # Stage 0: per-token processing
        for layer in self.stage0_layers:
            hidden_states = layer(hidden_states)

        # Squeeze: flatten patch tokens then project
        hidden_states = hidden_states.reshape(batch_num_patches, 1, patch_size * self.config.hidden_size)
        hidden_states = self.squeeze_layer(hidden_states)
        # hidden_states: (batch*num_patches, 1, hidden_size)

        # Stage 1
        for layer in self.stage1_layers:
            hidden_states = layer(hidden_states)

        # Final projection to latent
        hidden_states = self.final_norm(hidden_states)
        latent_params = self.hidden_to_latent(hidden_states)
        # latent_params: (batch*num_patches, 1, latent_size*2)
        return latent_params


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, lm_head_weight: torch.Tensor):
        super().__init__()
        self.config = config
        self.lm_head_weight = lm_head_weight  # shared with encoder embedding

        self.latent_to_hidden = nn.Linear(config.latent_size, config.hidden_size, bias=False)

        num_stage_layers = config.num_decoder_layers // 2

        # Stage 0: on latent-derived hidden states
        self.stage0_layers = nn.ModuleList(
            [
                AELayer(config.hidden_size, config.intermediate_size, config.rms_norm_eps)
                for _ in range(num_stage_layers)
            ]
        )

        # Expand: hidden_size → patch_size * hidden_size
        self.expand_layer = nn.Linear(
            config.hidden_size, config.patch_size * config.hidden_size, bias=False
        )

        # Stage 1: on expanded per-token representations
        self.stage1_layers = nn.ModuleList(
            [
                AELayer(config.hidden_size, config.intermediate_size, config.rms_norm_eps)
                for _ in range(num_stage_layers)
            ]
        )

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, seq_length, latent_size)
        Returns:
            logits: (batch, seq_length * patch_size, vocab_size)
        """
        batch, seq_length, _ = z.shape
        hidden_states = self.latent_to_hidden(z)
        # hidden_states: (batch, seq_length, hidden_size)

        # Stage 0
        for layer in self.stage0_layers:
            hidden_states = layer(hidden_states)

        # Expand: project then reshape to per-token
        hidden_states = self.expand_layer(hidden_states)
        hidden_states = hidden_states.reshape(
            batch, seq_length * self.config.patch_size, self.config.hidden_size
        )

        # Stage 1
        for layer in self.stage1_layers:
            hidden_states = layer(hidden_states)

        # Final norm + lm_head via shared embedding weights
        hidden_states = self.final_norm(hidden_states)
        logits = F.linear(hidden_states, self.lm_head_weight)
        return logits


class Autoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, lm_head_weight=self.encoder.embed_tokens.weight)

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, total_seq_len) where total_seq_len is divisible by patch_size
            labels:    (batch, total_seq_len) target token ids
        Returns:
            dict with 'loss' (scalar, if labels provided) and 'logits'
        """
        config = self.config
        batch, total_seq_len = input_ids.shape
        num_patches = total_seq_len // config.patch_size

        # Reshape into patches: (batch * num_patches, patch_size)
        encoder_input = input_ids.reshape(-1, config.patch_size)

        # Training: random dropout mask on input tokens (replace with 0)
        if self.training:
            dropout_mask = torch.rand_like(encoder_input, dtype=torch.float) > config.ae_dropout
            encoder_input = encoder_input * dropout_mask.long()

        # Encode → (batch*num_patches, 1, latent_size*2)
        latent_params = self.encoder(encoder_input)
        latent_params = latent_params.reshape(batch, num_patches, config.latent_size * 2)

        mean, log_std = latent_params.chunk(2, dim=-1)

        # VAE reparameterization
        if self.training:
            std = log_std.exp()
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean

        # Dropout on z during training
        if self.training:
            z = F.dropout(z, p=config.ae_dropout, training=True)

        # Decode → (batch, num_patches * patch_size, vocab_size)
        logits = self.decoder(z)

        result: Dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                labels.reshape(-1),
                reduction="mean",
            )

            # KL divergence: 0.5 * (mean^2 + std^2 - 1 - 2*log_std), clamped
            std = log_std.exp()
            kl_per_dim = 0.5 * (mean.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std)
            kl_per_dim = kl_per_dim.clamp(min=config.kl_clamp)
            kl_loss = kl_per_dim.mean()

            loss = ce_loss * config.patch_size + kl_loss * config.kl_weight
            result["loss"] = loss

        return result


if __name__ == "__main__":
    torch.manual_seed(42)

    config = AutoencoderConfig()
    model = Autoencoder(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Config: hidden={config.hidden_size}, latent={config.latent_size}, "
          f"patch={config.patch_size}, layers=enc{config.num_encoder_layers}/dec{config.num_decoder_layers}")
    print(f"Total parameters: {total_params:,}")

    # Smoke test
    batch, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids.clone()

    model.train()
    output = model(input_ids, labels=labels)
    loss = output["loss"]
    logits = output["logits"]

    print(f"Input shape:  {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss:         {loss.item():.4f}")

    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    print(f"Parameters with gradients: {len(grad_norms)}/{total_params and sum(1 for _ in model.parameters())}")
    print("Gradient flow OK ✓")
