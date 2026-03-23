# CALM-nanoGPT: Continuous Autoregressive Language Models on nanoGPT

[CALM](https://github.com/shaochenze/calm) (Continuous Autoregressive Language Models) を
[nanoGPT](https://github.com/karpathy/nanoGPT) スタイルで再実装したPoCプロジェクト。

HuggingFace依存を排除し、Pure PyTorch (~2000行) で CALM の全パイプラインを実装。

## アーキテクチャ

| コンポーネント | ファイル | 説明 |
|---|---|---|
| Autoencoder | `autoencoder.py` | VAE: Kトークン → 連続潜在ベクトル → Kトークン復元 |
| MLPGenerator | `mlp_generator.py` | ノイズ条件付きMLP生成ヘッド |
| CALM Model | `model.py` | RoPE Transformer + パッチ入力 + Energy Score損失 |
| AE訓練 | `train_autoencoder.py` | nanoGPTスタイルのAE訓練ループ |
| CALM訓練 | `train_calm.py` | 凍結AE + CALM本体の訓練ループ |

## クイックスタート

### 1. データ準備
```bash
python data/prepare_shakespeare.py
```

### 2. Autoencoder 訓練
```bash
# CPU (PoC)
python train_autoencoder.py config/train_ae_shakespeare.py

# GPU
python train_autoencoder.py --device=cuda --compile=True --dtype=bfloat16
```

### 3. CALM 本体訓練
```bash
# CPU (PoC)
python train_calm.py config/train_calm_shakespeare.py

# GPU
python train_calm.py --device=cuda --compile=True --dtype=bfloat16
```

## 設計方針

- **nanoGPTの哲学を踏襲**: 単一ファイル、HF非依存、configurator.pyによるシンプルな設定
- **LLaMAアーキテクチャ**: RoPE、GQA、SiLU-gated MLP、RMSNorm、Pre-norm
- **CALM固有機能**: パッチ入力圧縮、Energy Score損失、BrierLM的Temperature Sampling

## 元論文との対応

| 元CALM | 本実装 |
|---|---|
| `LlamaModel` (HF) | `model.py` の `Block` + `CausalSelfAttention` |
| `modeling_autoencoder.py` (HF) | `autoencoder.py` (standalone) |
| `modeling_energy.py` (HF) | `model.py` の `CALM.energy_score()` |
| `MLPGenerator` (HF) | `mlp_generator.py` (standalone) |
| HF `Trainer` | `train_autoencoder.py` / `train_calm.py` (nanoGPTスタイル) |

## 依存関係

```
torch>=2.0
numpy
tiktoken  # データ準備のみ
requests  # データ準備のみ
```
