# Config for training CALM on tiny Shakespeare (PoC)
# Usage: python train_calm.py config/train_calm_shakespeare.py

out_dir = 'out-calm'
ae_checkpoint = 'out-ae/ckpt.pt'
dataset = 'data'

batch_size = 8
block_size = 256
gradient_accumulation_steps = 1

# Small model for PoC
n_layer = 4
n_head = 4
n_kv_head = 4
n_embd = 128
intermediate_size = 256

# training
max_iters = 5000
eval_interval = 500
eval_iters = 100
log_interval = 50
learning_rate = 3e-4
warmup_iters = 200

# PoC-friendly settings (no GPU required)
device = 'cpu'
compile = False
dtype = 'float32'
num_samples = 4  # fewer samples for CPU
