"""
TRACE — exp09: Dropout2dを探索

exp08(weight_decay=1e-3)でgapは1.78->0.50まで縮んだが、まだtrain<valの余地はある。
ConvBlock末尾のDropout2dを何段階か試して、weight_decayと併用した効果を見る。
train.pyのCONFIG(DROPOUT_P以外、WEIGHT_DECAY=1e-3含む)は固定。
"""

import shutil
from pathlib import Path

import train

CANDIDATES = [0.0, 0.1, 0.2, 0.3]
HERE = Path(__file__).resolve().parent

results = {}
for p in CANDIDATES:
    train.DROPOUT_P = p
    train.CKPT_PATH = HERE / f"sweep_dropout_{p:.1f}.pt"
    print(f"\n===== dropout_p={p:.1f} =====")
    best_val, train_at_best = train.main()
    results[p] = (best_val, train_at_best)

print("\n===== summary =====")
for p, (best_val, train_at_best) in sorted(results.items(), key=lambda kv: kv[1][0]):
    gap = best_val - train_at_best
    print(f"dropout_p={p:.1f}  best_val={best_val:.6f}  train@best={train_at_best:.6f}  gap={gap:.6f}")

best_p = min(results, key=lambda k: results[k][0])
best_ckpt = HERE / f"sweep_dropout_{best_p:.1f}.pt"
shutil.copy(best_ckpt, HERE / "best_model.pt")
print(f"\nbest dropout_p = {best_p:.1f} (val={results[best_p][0]:.6f}) -> best_model.pt に反映")
