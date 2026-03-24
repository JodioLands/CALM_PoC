# CALM config for Google Colab — patch_size=4 (true CALM: 4-token simultaneous prediction)
# Requires substantial data/compute; best attempted with Colab Pro (A100) or large dataset.
# Usage: python train_calm.py config/train_calm_colab_k4.py

out_dir = 'out-calm-k4'
ae_checkpoint = 'out-ae/ckpt.pt'
dataset = 'data'

batch_size = 64
block_size = 256
gradient_accumulation_steps = 4  # effective batch = 256

# Model size
n_layer = 6
n_head = 8
n_kv_head = 8
n_embd = 256
intermediate_size = 512

# Training — needs more iters than AR due to harder objective
max_iters = 20000
eval_interval = 2000
eval_iters = 100
log_interval = 500
learning_rate = 3e-4
warmup_iters = 1000
decay_lr = True
min_lr = 3e-5

dropout = 0.1

# CALM-specific
loss_type = 'ce'
patch_size = 4

# GPU settings
device = 'cuda'
compile = True
dtype = 'bfloat16'
num_samples = 8
