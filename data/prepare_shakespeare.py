"""
Data preparation script for CALM-nanoGPT PoC using the tiny Shakespeare dataset.
Downloads, tokenizes (GPT-2 via tiktoken), and saves train/val splits as
numpy memmap files with patch-aligned token counts.
"""

import os
import pickle

import numpy as np

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required but not installed. Install it with:\n"
        "  pip install tiktoken"
    )

try:
    import requests
except ImportError:
    raise ImportError(
        "requests is required but not installed. Install it with:\n"
        "  pip install requests"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
PATCH_SIZE = 4
TRAIN_RATIO = 0.9

# All output files go next to this script
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(DATA_DIR, "input.txt")

# ---------------------------------------------------------------------------
# 1. Download dataset
# ---------------------------------------------------------------------------
if not os.path.exists(INPUT_PATH):
    print(f"Downloading tiny Shakespeare dataset from {DATA_URL} ...")
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()
    with open(INPUT_PATH, "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"Saved to {INPUT_PATH} ({len(resp.text):,} chars)")
else:
    print(f"Dataset already present at {INPUT_PATH}")

# ---------------------------------------------------------------------------
# 2. Read and tokenize
# ---------------------------------------------------------------------------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Raw text length: {len(text):,} characters")

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(text)

# ---------------------------------------------------------------------------
# 3. Trim to a multiple of patch_size
# ---------------------------------------------------------------------------
n = len(tokens)
n = n - (n % PATCH_SIZE)
tokens = tokens[:n]

# ---------------------------------------------------------------------------
# 4. Train / val split (90 / 10), patch-aligned
# ---------------------------------------------------------------------------
split_idx = int(n * TRAIN_RATIO)
split_idx = split_idx - (split_idx % PATCH_SIZE)  # ensure patch alignment

train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

# ---------------------------------------------------------------------------
# 5. Save as binary files (uint16 memmap-compatible)
# ---------------------------------------------------------------------------
train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)

train_path = os.path.join(DATA_DIR, "train.bin")
val_path = os.path.join(DATA_DIR, "val.bin")

train_ids.tofile(train_path)
val_ids.tofile(val_path)

# ---------------------------------------------------------------------------
# 6. Save metadata
# ---------------------------------------------------------------------------
meta = {
    "vocab_size": enc.n_vocab,
    "patch_size": PATCH_SIZE,
    "train_size": len(train_tokens),
    "val_size": len(val_tokens),
    "encoding": "gpt2",
}
meta_path = os.path.join(DATA_DIR, "meta.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)

# ---------------------------------------------------------------------------
# 7. Print statistics
# ---------------------------------------------------------------------------
print("--- Shakespeare dataset ready ---")
print(f"Total tokens (patch-aligned): {n:,}")
print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")
print(f"Vocab size:   {enc.n_vocab:,}")
print(f"Patch size:   {PATCH_SIZE}")
print(f"Files written: {train_path}, {val_path}, {meta_path}")
