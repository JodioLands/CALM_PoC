# AE config for Google Colab (T4/A100 GPU)
# Usage: python train_autoencoder.py config/train_ae_colab.py

out_dir = 'out-ae'
dataset = 'data'

# GPU: much larger batch possible
batch_size = 128
block_size = 256
gradient_accumulation_steps = 1

# Training — 3000 iters enough on GPU
max_iters = 3000
eval_interval = 500
eval_iters = 50
log_interval = 100
learning_rate = 1e-3
warmup_iters = 200

# GPU settings
device = 'cuda'
compile = True
dtype = 'bfloat16'
