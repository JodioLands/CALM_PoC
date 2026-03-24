"""
CALM — Continuous Autoregressive Language Model.

Main model integrating a nanoGPT-style Transformer backbone with
patch-based continuous prediction via an MLPGenerator and a frozen Autoencoder.

No HuggingFace dependencies.  Uses only torch, torch.nn, and local imports.
"""

import math
import random
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


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


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for Rotary Position Embeddings."""

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """Return (cos, sin) each of shape ``(T, head_dim)``."""
        freqs = torch.outer(position_ids.float(), self.inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    """Apply RoPE to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head attention with RoPE and optional GQA.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` with
    ``is_causal=True`` for efficient causal masking (flash attention when
    available).
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        )
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(
            config.n_embd, self.n_head * self.head_dim, bias=config.bias
        )
        self.k_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias
        )
        self.o_proj = nn.Linear(
            self.n_head * self.head_dim, config.n_embd, bias=config.bias
        )

        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.block_size, config.rope_theta
        )

    def forward(self, x, position_ids=None, past_kv=None):
        B, T, C = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        cos, sin = self.rotary_emb(x, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand k, v to match the number of query heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Efficient causal attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y


class GatedMLP(nn.Module):
    """SiLU-gated MLP (LLaMA-style): ``down_proj(silu(gate_proj(x)) * up_proj(x))``"""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.up_proj = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm Transformer block (RMSNorm → Attention, RMSNorm → MLP)."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(self, x, position_ids=None, past_kv=None):
        h = x + self.attn(self.ln_1(x), position_ids, past_kv)
        h = h + self.mlp(self.ln_2(h))
        return h


# ---------------------------------------------------------------------------
# CALM configuration and model
# ---------------------------------------------------------------------------


@dataclass
class CALMConfig:
    # Transformer
    block_size: int = 2048  # max sequence length in *patches*
    vocab_size: int = 50304
    n_layer: int = 16
    n_head: int = 16
    n_kv_head: int = 16  # for GQA; None → same as n_head
    n_embd: int = 1024
    intermediate_size: int = 2752
    dropout: float = 0.0
    bias: bool = False
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # CALM-specific
    patch_size: int = 4
    latent_size: int = 128
    noise_size: int = 64
    num_mlp_layers: int = 4  # MLPGenerator depth
    num_samples: int = 8  # samples drawn for energy-score loss
    beta: float = 1.0  # energy-score distance exponent
    loss_type: str = "energy"  # "energy" or "mse"


class CALM(nn.Module):
    """Continuous Autoregressive Language Model.

    Replaces nanoGPT's per-token logit prediction with patch-level
    continuous latent prediction via an MLPGenerator, supervised by a
    frozen VAE Autoencoder through an energy-score loss.
    """

    def __init__(self, config: CALMConfig, ae_model=None):
        super().__init__()
        self.config = config

        # ---- Transformer backbone ----
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.n_embd // config.n_head, config.block_size, config.rope_theta
        )

        # ---- CALM: patch projection ----
        # Maps K concatenated token embeddings → single hidden vector
        self.embed_proj = nn.Sequential(
            nn.Linear(config.patch_size * config.n_embd, 2 * config.n_embd),
            nn.SiLU(),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.LayerNorm(config.n_embd, eps=config.rms_norm_eps),
        )

        # ---- MLPGenerator (generative head) ----
        from mlp_generator import MLPGenerator, MLPGeneratorConfig

        gen_config = MLPGeneratorConfig(
            hidden_size=config.n_embd,
            latent_size=config.latent_size,
            noise_size=config.noise_size,
            num_mlp_layers=config.num_mlp_layers,
        )
        self.generator = MLPGenerator(gen_config)

        # ---- Direct latent prediction head (MSE mode) ----
        self.latent_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.latent_size),
        )

        # ---- Direct token prediction head (CE mode) ----
        # Expand hidden → K positions, then shared vocab projection
        self.token_expand = nn.Sequential(
            nn.Linear(config.n_embd, config.patch_size * config.n_embd),
            nn.SiLU(),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: lm_head shares weights with wte
        self.lm_head.weight = self.wte.weight

        # ---- Frozen autoencoder (may be set later) ----
        self.ae_model = ae_model

        # ---- Weight initialisation ----
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 trick)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        # Zero the generator's final layer for a stable start
        self.generator.initialize_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Energy-score loss
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(x1, x2, beta):
        return torch.pow(torch.linalg.norm(x1 - x2, ord=2, dim=-1), beta)

    def energy_score(self, predictions, target_mean, target_log_std, beta=1.0):
        """Compute the energy score between predicted and target distributions.

        Args:
            predictions:    (num_samples, total_patches, latent_size)
            target_mean:    (total_patches, latent_size)
            target_log_std: (total_patches, latent_size)
        Returns:
            Per-patch energy score — shape ``(total_patches,)``.
        """
        n_x = predictions.shape[0]

        # E[||X - X'||^β]  (prediction ↔ prediction, excluding diagonal)
        x_i = predictions.unsqueeze(1)  # (n_x, 1,   P, L)
        x_j = predictions.unsqueeze(0)  # (1,   n_x, P, L)
        dist_xx = (
            self._distance(x_i, x_j, beta).sum(dim=(0, 1)) / (n_x * (n_x - 1))
        )

        # E[||X - Y||^β]  (prediction ↔ samples from target distribution)
        std = torch.exp(target_log_std)
        n_y = 100
        eps = torch.randn((n_y, *target_mean.shape), device=target_mean.device)
        y = target_mean + eps * std  # (n_y, P, L)

        x_ = predictions.unsqueeze(1)  # (n_x, 1,   P, L)
        y_ = y.unsqueeze(0)  # (1,   n_y, P, L)
        dist_xy = self._distance(x_, y_, beta).mean(dim=(0, 1))

        return dist_xx - 2 * dist_xy

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch, seq_len) — ``seq_len`` must be a multiple of
                       ``patch_size``.
            targets:   (batch, seq_len) — same shape; shifted by one patch
                       internally.  ``None`` for inference.

        Returns:
            *Training*:  ``(loss, latent_predictions)``
            *Inference*: ``(batch, n_embd)`` hidden state of the last patch.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        patch_size = self.config.patch_size
        assert seq_len % patch_size == 0, (
            f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})"
        )
        num_patches = seq_len // patch_size

        if targets is not None:
            # ===== Training / Evaluation =====
            # Target: tokens shifted by one patch
            target_tokens = targets[:, patch_size:]

            # Embed tokens → patch vectors
            tok_emb = self.wte(input_ids)  # (B, S, E)
            tok_emb = tok_emb.reshape(
                batch_size, num_patches, patch_size * self.config.n_embd
            )
            # Drop last patch (no target for it)
            x = self.embed_proj(tok_emb[:, :-1, :])  # (B, P-1, E)

            position_ids = torch.arange(
                x.shape[1], dtype=torch.long, device=device
            )

            # Transformer
            for block in self.blocks:
                x = block(x, position_ids=position_ids)
            x = self.ln_f(x)  # (B, P-1, E)

            if self.config.loss_type == "ce":
                # Direct next-patch token prediction with CE loss
                expanded = self.token_expand(x)  # (B, P-1, K*E)
                expanded = expanded.reshape(
                    batch_size * (num_patches - 1), patch_size, self.config.n_embd
                )
                logits = self.lm_head(expanded)  # (B*(P-1), K, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    target_tokens.reshape(-1),
                )
                return loss, logits.detach()

            # Latent-based modes need AE encoder
            with torch.no_grad():
                ae_input = target_tokens.reshape(-1, patch_size)
                latent_states = self.ae_model.encoder(ae_input)
                latent_states = latent_states.reshape(
                    batch_size, num_patches - 1, self.config.latent_size * 2
                )
                target_mean, target_log_std = torch.chunk(latent_states, 2, dim=-1)

            if self.config.loss_type == "mse":
                # Direct latent prediction with MSE loss
                pred_latent = self.latent_head(x)  # (B, P-1, latent_size)
                loss = F.mse_loss(pred_latent, target_mean)
                return loss, pred_latent.detach()
            elif self.config.loss_type == "hybrid":
                # MSE on latents + CE on decoded tokens
                pred_latent = self.latent_head(x)  # (B, P-1, latent_size)
                mse_loss = F.mse_loss(pred_latent, target_mean)

                # Decode predicted latents through frozen AE decoder
                # (AE params are frozen but gradients flow through to latent_head)
                pred_3d = pred_latent.reshape(
                    batch_size * (num_patches - 1), 1, self.config.latent_size
                )
                logits = self.ae_model.decoder(pred_3d)
                # logits: (B*(P-1), patch_size, vocab_size)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )
                loss = mse_loss + ce_loss
                return loss, pred_latent.detach()
            else:
                # MLPGenerator → multiple latent samples + Energy Score
                hidden_flat = x.reshape(-1, self.config.n_embd)
                hidden_repeated = hidden_flat.unsqueeze(0).expand(
                    self.config.num_samples, -1, -1
                )
                latent_predictions = self.generator.sample(hidden_repeated)

                target_mean_flat = target_mean.reshape(-1, self.config.latent_size)
                target_log_std_flat = target_log_std.reshape(-1, self.config.latent_size)
                loss = -self.energy_score(
                    latent_predictions,
                    target_mean_flat,
                    target_log_std_flat,
                    self.config.beta,
                )
                loss = loss.mean()
                return loss, latent_predictions.detach()

        else:
            # ===== Inference =====
            tok_emb = self.wte(input_ids)
            tok_emb = tok_emb.reshape(
                batch_size, num_patches, patch_size * self.config.n_embd
            )
            x = self.embed_proj(tok_emb)

            position_ids = torch.arange(
                num_patches, dtype=torch.long, device=device
            )

            for block in self.blocks:
                x = block(x, position_ids=position_ids)
            x = self.ln_f(x)

            return x[:, -1, :]  # (B, E) — last patch hidden state

    # ------------------------------------------------------------------
    # Optimizer (nanoGPT-style)
    # ------------------------------------------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Create AdamW with weight-decay / no-decay param groups."""
        import inspect

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_patches: int = 100,
        temperature: float = 0.5,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_samples: int = 200,
    ):
        """Autoregressively generate token patches.

        Args:
            input_ids:          (batch, prompt_len) — padded to patch boundary
            max_new_patches:    number of patches to generate.
            temperature:        sampling temperature (0 = greedy).
            top_k:              top-k filtering (0 = disabled).
            top_p:              nucleus sampling threshold (1.0 = disabled).
            repetition_penalty: penalize already-generated tokens (1.0 = none).
            num_samples:        candidates drawn for energy mode.
        Returns:
            (batch, total_tokens) including the prompt.
        """
        self.eval()
        if self.config.loss_type not in ("ce",):
            assert self.ae_model is not None, "Autoencoder must be set for generation"
        patch_size = self.config.patch_size
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Pad prompt to a multiple of patch_size
        prompt_len = input_ids.shape[1]
        if prompt_len % patch_size != 0:
            pad_len = patch_size - (prompt_len % patch_size)
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.zeros(
                        batch_size, pad_len, dtype=torch.long, device=device
                    ),
                ],
                dim=1,
            )

        all_tokens = input_ids.clone()

        for _ in range(max_new_patches):
            hidden = self.forward(all_tokens)  # (B, E)

            if self.config.loss_type == "ce":
                # CE mode: direct token prediction with sampling
                expanded = self.token_expand(hidden)  # (B, K*E)
                expanded = expanded.reshape(batch_size, patch_size, self.config.n_embd)
                logits = self.lm_head(expanded)  # (B, K, V)

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        prev = all_tokens[b].unique()
                        for k in range(patch_size):
                            score = logits[b, k, prev]
                            logits[b, k, prev] = torch.where(
                                score > 0, score / repetition_penalty,
                                score * repetition_penalty)

                if temperature <= 0:
                    next_tokens = torch.argmax(logits, dim=-1)
                else:
                    logits = logits / temperature
                    # Top-k filtering
                    if top_k > 0:
                        tk = min(top_k, logits.size(-1))
                        v, _ = torch.topk(logits, tk, dim=-1)
                        logits[logits < v[..., -1:]] = -float('inf')
                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                        sorted_logits[remove] = -float('inf')
                        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
                    probs = torch.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(
                        probs.reshape(-1, probs.size(-1)), num_samples=1
                    ).reshape(batch_size, patch_size)
            elif self.config.loss_type in ("mse", "hybrid"):
                # MSE/hybrid mode: direct latent prediction → AE decode
                latent = self.latent_head(hidden)  # (B, latent_size)
                logits = self.ae_model.decoder(
                    latent.unsqueeze(1)
                )  # (B, patch_size, V)
                next_tokens = torch.argmax(logits, dim=-1)
            elif temperature == 1.0:
                # Multi-sample then vote (BrierLM-style temperature)
                hidden_rep = hidden.unsqueeze(1).expand(
                    -1, num_samples, -1
                )  # (B, N, E)
                latents = self.generator.sample(hidden_rep)  # (B, N, latent)
                b, n, ls = latents.shape
                logits = self.ae_model.decoder(latents.reshape(b * n, 1, ls))
                # (B*N, patch_size, V) → (B, N, patch_size, V)
                logits = logits.reshape(b, n, patch_size, -1)
                candidates = torch.argmax(logits, dim=-1)  # (B, N, patch_size)

                inv_temp = max(int(round(1.0 / temperature)), 1)
                next_tokens_list = []
                for i in range(batch_size):
                    samples = [
                        tuple(candidates[i, j].tolist()) for j in range(n)
                    ]
                    counts = Counter(samples)
                    selected = None
                    for n_thr in range(inv_temp, 0, -1):
                        cands = {
                            p: c for p, c in counts.items() if c >= n_thr
                        }
                        if cands:
                            weights = [
                                math.comb(c, n_thr) for c in cands.values()
                            ]
                            selected = random.choices(
                                list(cands.keys()), weights=weights, k=1
                            )[0]
                            break
                    next_tokens_list.append(
                        torch.tensor(selected, dtype=torch.long, device=device)
                    )
                next_tokens = torch.stack(next_tokens_list)

            all_tokens = torch.cat([all_tokens, next_tokens], dim=1)

        return all_tokens


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from autoencoder import Autoencoder, AutoencoderConfig

    torch.manual_seed(42)

    # 1. Configs --------------------------------------------------------
    calm_cfg = CALMConfig(
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        intermediate_size=256,
        vocab_size=50304,
        block_size=512,
    )
    ae_cfg = AutoencoderConfig(vocab_size=50304)

    # 2. Frozen autoencoder --------------------------------------------
    ae = Autoencoder(ae_cfg)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # 3. CALM model -----------------------------------------------------
    model = CALM(calm_cfg, ae_model=ae)
    total = sum(p.numel() for p in model.parameters())
    trainable = model.get_num_params(non_embedding=True)
    print(f"Total params: {total:,}  |  Trainable (non-emb): {trainable:,}")

    # 4. Forward + backward ---------------------------------------------
    batch, seq_len = 2, 32
    ids = torch.randint(0, calm_cfg.vocab_size, (batch, seq_len))
    model.train()
    loss, preds = model(ids, targets=ids)
    print(f"Loss: {loss.item():.4f}  |  Predictions shape: {tuple(preds.shape)}")

    loss.backward()
    grads = {
        n: p.grad.norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    print(f"Params with gradients: {len(grads)}")
    assert len(grads) > 0, "No gradients!"
    print("Gradient flow OK ✓")

    # 5. Generate -------------------------------------------------------
    prompt = torch.randint(0, calm_cfg.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_patches=3, temperature=1.0)
    print(f"Generate: prompt {tuple(prompt.shape)} → output {tuple(generated.shape)}")

    print("All checks passed ✓")
