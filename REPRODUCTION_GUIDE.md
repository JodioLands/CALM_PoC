# 再現手順ガイド — CALM-nanoGPT PoC

Model A（CALMバックボーン、patch_size=1）と同じ生成結果を自分で再現するための手順書です。

---

## 必要な環境

| 項目 | バージョン |
|---|---|
| Python | 3.10 以上 |
| PyTorch | 2.0 以上（CPUのみでも動作） |
| tiktoken | 0.7 以上 |
| numpy | 1.24 以上 |

```bash
pip install torch tiktoken numpy
```

CPUのみ環境（GPU不要）で動作確認済みです。  
所要時間の目安：**AE学習 〜30分 + CALM学習 〜50分 = 合計 〜80分**（MacBook CPU基準）

---

## ディレクトリ構成

```
CALM_PoC/
├── autoencoder.py         # VAE Autoencoder（AE）モデル定義
├── mlp_generator.py       # MLPGenerator（Energy Scoreモード用）
├── model.py               # CALMモデル本体
├── train_autoencoder.py   # AE学習スクリプト
├── train_calm.py          # CALM学習スクリプト
├── configurator.py        # 設定ファイル上書きユーティリティ
├── config/
│   ├── train_ae_shakespeare.py     # AE用設定
│   └── train_calm_shakespeare.py  # CALM用設定
└── data/
    └── prepare_shakespeare.py      # データ準備スクリプト
```

---

## Step 1: データ準備

```bash
cd CALM_PoC
python3 data/prepare_shakespeare.py
```

**確認**：以下のファイルが生成されれば成功です。

```
data/train.bin   # 約 608 KB (304,000 トークン)
data/val.bin     # 約 68 KB  (34,000 トークン)
```

---

## Step 2: Autoencoder の学習

AEは4トークンをまとめて128次元の潜在変数に圧縮・復元するVAEです。

```bash
python3 train_autoencoder.py config/train_ae_shakespeare.py \
  --learning_rate=1e-3 \
  --max_iters=1000 \
  --eval_interval=500 \
  --eval_iters=50 \
  --batch_size=16
```

**ログの見方**：

```
step 500: train loss 0.63, val loss 0.63    ← val_loss が 1.0 以下なら良好
step 1000: train loss 0.66, val loss 0.66
```

**完了確認**：

```
out-ae/ckpt.pt   # 約 410 MB
```

**再構成精度の確認**（オプション）：

```bash
python3 -c "
import torch, tiktoken
from autoencoder import AutoencoderConfig, Autoencoder

ckpt = torch.load('out-ae/ckpt.pt', map_location='cpu', weights_only=False)
ae = Autoencoder(AutoencoderConfig(**ckpt['ae_config']))
ae.load_state_dict(ckpt['model'])
ae.eval()

enc = tiktoken.get_encoding('gpt2')
text = 'To be, or not to be, that is the question:'
tokens = enc.encode(text)

# 4トークン単位で整形
pad = (4 - len(tokens) % 4) % 4
tokens += [enc.eot_token] * pad
x = torch.tensor(tokens, dtype=torch.long).reshape(1, -1)
with torch.no_grad():
    result = ae(x)
    pred = result['logits'].argmax(-1).reshape(-1).tolist()
print('Original:', enc.decode(tokens))
print('Decoded: ', enc.decode(pred[:len(tokens)]))
"
```

97% 以上のトークン一致率が出れば OK です。

---

## Step 3: CALM モデルの学習（patch_size=1, ARモード）

AEを凍結した状態でCALMバックボーンを学習します。  
`loss_type=ce`（Cross Entropy）＋`patch_size=1`でシンプルな自己回帰モデルとして動作します。

```bash
python3 train_calm.py config/train_calm_shakespeare.py \
  --learning_rate=1e-3 \
  --warmup_iters=200 \
  --max_iters=5000 \
  --log_interval=200 \
  --eval_interval=1000 \
  --eval_iters=20 \
  --loss_type=ce \
  --out_dir=out-calm-ar \
  --dropout=0.1 \
  --batch_size=8 \
  --patch_size=1 \
  --block_size=256 \
  --decay_lr=True \
  --min_lr=1e-4
```

**ログの見方**：

```
iter 200:  loss 6.03  ← 最初は高い
iter 500:  loss 4.90  ← 急速に低下
iter 1000: val loss 5.03  ← val が 5.5 以下で良好な学習
iter 2000: val loss 4.96  ← ベスト (best_val_loss として保存)
iter 5000: val loss 5.28  ← わずかに過学習するが問題なし
```

**完了確認**：

```
out-calm-ar/ckpt.pt   # 約 220 MB（best_val_loss のチェックポイント）
```

---

## Step 4: テキスト生成

```bash
python3 -c "
import torch, tiktoken
from autoencoder import AutoencoderConfig, Autoencoder
from model import CALM, CALMConfig

# AE の読み込み
ae_ckpt = torch.load('out-ae/ckpt.pt', map_location='cpu', weights_only=False)
ae = Autoencoder(AutoencoderConfig(**ae_ckpt['ae_config']))
ae.load_state_dict(ae_ckpt['model'])
ae.eval()

# CALM モデルの読み込み
calm_ckpt = torch.load('out-calm-ar/ckpt.pt', map_location='cpu', weights_only=False)
calm = CALM(CALMConfig(**calm_ckpt['model_args']), ae)
calm.load_state_dict(calm_ckpt['model'], strict=False)
calm.eval()

enc = tiktoken.get_encoding('gpt2')

# --- 生成 ---
prompt = 'ROMEO:\nO, she doth teach the torches to burn bright!'
tokens = enc.encode(prompt)
x = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    gen = calm.generate(
        x,
        max_new_patches=120,  # 生成トークン数
        temperature=0.75,     # 0.5〜1.0 がおすすめ
        top_k=40,             # 上位 40 トークンからサンプリング
        repetition_penalty=1.2,  # 繰り返し抑制
    )

print(enc.decode(gen[0].tolist()))
"
```

**期待される出力例**：

```
ROMEO:
O, she doth teach the torches to burn bright!
That I do know some of death's wife's death,
So soon as it is so that you should have:
I fear, for his mother, but not my heart.

FRIAR LAURENCE:
Away with me, and be the day's death,
I'll make thee see me all that thou hast...
```

---

## パラメータ調整ガイド

生成品質は以下のパラメータで調整できます：

| パラメータ | 推奨範囲 | 効果 |
|---|---|---|
| `temperature` | 0.6〜0.9 | 低い → 保守的・繰り返しあり / 高い → 多様・時にランダム |
| `top_k` | 20〜100 | 低い → 安定 / 高い → 多様 |
| `repetition_penalty` | 1.1〜1.5 | 高いほど同じ単語の繰り返しを抑制 |
| `max_new_patches` | 50〜200 | 生成する追加トークン数 |

---

## トラブルシューティング

### `out-ae/ckpt.pt` が見つからない
→ Step 2 のAE学習が完了していません。学習を先に実行してください。

### 生成テキストが意味不明（ランダムな単語の羅列）
→ `val_loss` が 5.5 以下になっているか確認してください。  
→ `temperature` を 0.7 以下に下げてみてください。  
→ `max_iters=5000` まで学習が完了しているか確認してください。

### メモリ不足
→ `batch_size=4` に下げてください。  
→ `block_size=128` に下げてください。

### 学習が遅い（CPUの場合）
→ `eval_iters=10` に下げると評価が速くなります。  
→ Apple Silicon Mac の場合は `--device=mps` を試せますが、この規模では CPU の方が速いです。

---

## 参考：各モデルの比較

| モデル | patch_size | val_loss | 生成品質 | 備考 |
|---|---|---|---|---|
| **Model A（本ガイド）** | 1 | **4.96** | ✅ 意味のある文章 | CALMバックボーン＋通常AR |
| Model B (CALM本来) | 4 | 5.78 | ⚠️ ワードサラダ | 小規模データでは困難 |

CALM 本来の多トークン同時予測（patch_size=4 + Energy Score）は、
数十億トークン規模のデータと GPU が必要です。
このPoC では `patch_size=1` の標準ARモードで意味のある文章生成を達成しました。
