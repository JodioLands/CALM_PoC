# Config for training autoencoder on tiny Shakespeare
# Usage: python train_autoencoder.py config/train_ae_shakespeare.py

out_dir = 'out-ae'
dataset = 'data'

# small batch for PoC
batch_size = 32
block_size = 256  # must be multiple of patch_size (4)
gradient_accumulation_steps = 1

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
