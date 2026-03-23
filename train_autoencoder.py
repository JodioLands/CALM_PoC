"""
Training script for the CALM Autoencoder (VAE-based token reconstructor).

Follows the nanoGPT train.py pattern: single-file, DDP support, gradient
accumulation, cosine LR decay, wandb logging, checkpointing.

Usage:
    python train_autoencoder.py                              # defaults
    python train_autoencoder.py config/train_ae_shakespeare.py  # config override
    python train_autoencoder.py --batch_size=32 --device=cpu    # CLI overrides
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

# -----------------------------------------------------------------------------
# default config values
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out-ae'
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'calm-ae'
wandb_run_name = 'ae'

# data
dataset = 'data'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 512  # total tokens per sample (must be multiple of patch_size)

# autoencoder model config
patch_size = 4
vocab_size = 50304
hidden_size = 512
intermediate_size = 1280
num_encoder_layers = 2
num_decoder_layers = 2
latent_size = 128
ae_dropout = 0.15
kl_clamp = 0.5
kl_weight = 1e-3

# optimizer
learning_rate = 3e-4
max_iters = 30000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False  # CALM uses constant LR for AE
warmup_iters = 1000
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
    # For autoencoder: input == labels (reconstruction target)
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x

# -----------------------------------------------------------------------------
# model init
# -----------------------------------------------------------------------------
ae_config_args = dict(
    patch_size=patch_size,
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    latent_size=latent_size,
    ae_dropout=ae_dropout,
    kl_clamp=kl_clamp,
    kl_weight=kl_weight,
)

iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    if master_process:
        print("Initializing a new autoencoder from scratch")
    ae_conf = AutoencoderConfig(**ae_config_args)
    model = Autoencoder(ae_conf)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # restore model config from checkpoint
    checkpoint_ae_config = checkpoint['ae_config']
    ae_conf = AutoencoderConfig(**checkpoint_ae_config)
    model = Autoencoder(ae_conf)
    state_dict = checkpoint['model']
    # fix DDP key prefix if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.to(device)

# print parameter count
total_params = sum(p.numel() for p in model.parameters())
if master_process:
    print(f"Autoencoder parameters: {total_params:,}")
    print(f"  hidden_size={ae_conf.hidden_size}, latent_size={ae_conf.latent_size}, "
          f"patch_size={ae_conf.patch_size}, "
          f"encoder_layers={ae_conf.num_encoder_layers}, decoder_layers={ae_conf.num_decoder_layers}")

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
            x = get_batch(split)
            with ctx:
                output = model(input_ids=x, labels=x)
            losses[k] = output['loss'].item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
x = get_batch('train')  # prefetch first batch
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
                    'ae_config': ae_config_args,
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
            output = model(input_ids=x, labels=x)
            loss = output['loss'] / gradient_accumulation_steps
        # prefetch next batch while backward runs
        x = get_batch('train')
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
