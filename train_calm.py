import sys
sys.stdout.reconfigure(line_buffering=True)
"""
Training script for the CALM (Continuous Autoregressive Language Model).

Follows the nanoGPT train.py pattern: single-file, DDP support, gradient
accumulation, cosine LR decay, wandb logging, checkpointing.

Trains a CALM model with a frozen pre-trained Autoencoder providing
ground-truth latent targets for the energy-score loss.

Usage:
    python train_calm.py                                    # defaults
    python train_calm.py config/train_calm_shakespeare.py   # config override
    python train_calm.py --batch_size=32 --device=cpu       # CLI overrides
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from autoencoder import AutoencoderConfig, Autoencoder
from model import CALMConfig, CALM

# -----------------------------------------------------------------------------
# default config values
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out-calm'
eval_interval = 1000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'calm'
wandb_run_name = 'calm-energy'

# data
dataset = 'data'
gradient_accumulation_steps = 8
batch_size = 8
block_size = 512  # total tokens per sample (must be multiple of patch_size)

# autoencoder (frozen)
ae_checkpoint = 'out-ae/ckpt.pt'  # path to pre-trained AE checkpoint

# CALM model config
patch_size = 4
vocab_size = 50304
n_layer = 16
n_head = 16
n_kv_head = 16
n_embd = 1024
intermediate_size = 2752
dropout = 0.0
bias = False
latent_size = 128
noise_size = 64
num_mlp_layers = 4
num_samples = 8
beta = 1.0
loss_type = 'energy'  # 'energy' or 'mse'

# optimizer
learning_rate = 3e-4
max_iters = 250000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False  # CALM uses constant LR
warmup_iters = 2000
min_lr = 3e-5  # minimum learning rate (used if decay_lr=True)

# system
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# -----------------------------------------------------------------------------
# poor man's configurator — grab config keys before exec
# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith('_') and isinstance(v, (int, float, bool, str))
]
exec(open('configurator.py').read())  # overrides from file / CLI
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# validation
# -----------------------------------------------------------------------------
assert block_size % patch_size == 0, \
    f"block_size ({block_size}) must be a multiple of patch_size ({patch_size})"

if not os.path.exists(ae_checkpoint):
    raise FileNotFoundError(
        f"Autoencoder checkpoint not found at '{ae_checkpoint}'. "
        f"Train the autoencoder first with train_autoencoder.py, or set "
        f"--ae_checkpoint=<path> to the correct checkpoint path."
    )

# -----------------------------------------------------------------------------
# DDP setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# device detection
device_type = 'cuda' if 'cuda' in device else device
if device_type == 'cuda' and not torch.cuda.is_available():
    if torch.backends.mps.is_available():
        device = 'mps'
        device_type = 'mps'
    else:
        device = 'cpu'
        device_type = 'cpu'
    if master_process:
        print(f"CUDA not available, falling back to device: {device}")

# mixed precision context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type in ('cpu', 'mps') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset)


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    # y is same as x; the CALM model handles the internal shift via targets[:, patch_size:]
    y = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# load frozen autoencoder
# -----------------------------------------------------------------------------
if master_process:
    print(f"Loading autoencoder from {ae_checkpoint}")
ae_ckpt = torch.load(ae_checkpoint, map_location=device, weights_only=False)
ae_config = ae_ckpt['ae_config']
if isinstance(ae_config, AutoencoderConfig):
    ae_model = Autoencoder(ae_config)
elif isinstance(ae_config, dict):
    ae_model = Autoencoder(AutoencoderConfig(**ae_config))
else:
    ae_model = Autoencoder(AutoencoderConfig(**vars(ae_config)))
state_dict = ae_ckpt['model']
# fix DDP / compile key prefix if needed
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
ae_model.load_state_dict(state_dict)
ae_model.to(device)
ae_model.eval()
for param in ae_model.parameters():
    param.requires_grad = False
if master_process:
    ae_params = sum(p.numel() for p in ae_model.parameters())
    print(f"Autoencoder loaded ({ae_params:,} params, frozen)")

# -----------------------------------------------------------------------------
# CALM model init
# -----------------------------------------------------------------------------
model_args = dict(
    block_size=block_size // patch_size,  # block_size in patches
    vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
    n_kv_head=n_kv_head, n_embd=n_embd, intermediate_size=intermediate_size,
    dropout=dropout, bias=bias, patch_size=patch_size,
    latent_size=latent_size, noise_size=noise_size,
    num_mlp_layers=num_mlp_layers, num_samples=num_samples, beta=beta,
    loss_type=loss_type,
)

iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    if master_process:
        print("Initializing a new CALM model from scratch")
    calm_config = CALMConfig(**model_args)
    model = CALM(calm_config, ae_model=ae_model)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming CALM training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    calm_config = CALMConfig(**checkpoint_model_args)
    model = CALM(calm_config, ae_model=ae_model)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.to(device)

# print parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if master_process:
    print(f"CALM trainable parameters: {total_params:,}")
    print(f"  n_layer={calm_config.n_layer}, n_head={calm_config.n_head}, "
          f"n_embd={calm_config.n_embd}, patch_size={calm_config.patch_size}, "
          f"latent_size={calm_config.latent_size}")

# -----------------------------------------------------------------------------
# optimizer
# -----------------------------------------------------------------------------
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type) \
    if hasattr(model, 'configure_optimizers') else \
    torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2),
                      weight_decay=weight_decay)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free memory

# compile
if compile:
    if master_process:
        print("Compiling the model... (takes a ~minute)")
    try:
        model = torch.compile(model)
    except Exception as e:
        if master_process:
            print(f"torch.compile failed ({e}), continuing without compilation")
        compile = False

# wrap DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# learning rate scheduler (cosine with warmup)
# -----------------------------------------------------------------------------
def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # constant LR (no decay)
    if not decay_lr:
        return learning_rate
    # cosine decay down to min_lr
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
x, y = get_batch('train')  # prefetch first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # set learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluation & checkpointing
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(losses['val'], best_val_loss)
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'ae_checkpoint': ae_checkpoint,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if master_process:
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward + backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            loss, _ = model(x, targets=y)
            loss = loss / gradient_accumulation_steps
        # prefetch next batch while backward runs
        x, y = get_batch('train')
        scaler.scale(loss).backward()

    # gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
