# CALM config for Google Colab — patch_size=1 (AR baseline, generates good text)
# Usage: python train_calm.py config/train_calm_colab_ar.py

out_dir = 'out-calm-ar'
ae_checkpoint = 'out-ae/ckpt.pt'
dataset = 'data'

# GPU: larger batch, bigger model
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2  # effective batch = 128

# Bigger model than CPU PoC
n_layer = 6
n_head = 8
n_kv_head = 8
n_embd = 256
intermediate_size = 512

# Training
max_iters = 10000
eval_interval = 1000
eval_iters = 100
log_interval = 200
learning_rate = 1e-3
warmup_iters = 500
decay_lr = True
min_lr = 1e-4

# Regularization
dropout = 0.1

# CALM-specific
loss_type = 'ce'
patch_size = 1

# GPU settings
device = 'cuda'
compile = True
dtype = 'bfloat16'
num_samples = 4
