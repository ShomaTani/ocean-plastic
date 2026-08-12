"""
TRACE — bexp03: 逆モデル用のLR再探索

forwardのベスト値(LR=3e-3)をそのまま流用しているが、逆モデルは問題の構造が
違う(bexp01/02参照)ので、最適なLRも違う可能性がある。forward/sweep_lr.pyと
同じやり方でtrain.pyのCONFIG(LR以外)を固定して複数候補を比較する。
"""

import shutil
from pathlib import Path

import train

CANDIDATES = [3e-4, 1e-3, 3e-3, 1e-2]
HERE = Path(__file__).resolve().parent

results = {}
for lr in CANDIDATES:
    train.LR = lr
    train.CKPT_PATH = HERE / f"sweep_lr_{lr:.0e}.pt"
    print(f"\n===== LR={lr:.0e} =====")
    best_val, train_at_best = train.main()
    results[lr] = (best_val, train_at_best)

print("\n===== summary =====")
for lr, (best_val, train_at_best) in sorted(results.items(), key=lambda kv: kv[1][0]):
    gap = best_val - train_at_best
    print(f"LR={lr:.0e}  best_val={best_val:.6f}  train@best={train_at_best:.6f}  gap={gap:.6f}")

best_lr = min(results, key=lambda k: results[k][0])
best_ckpt = HERE / f"sweep_lr_{best_lr:.0e}.pt"
shutil.copy(best_ckpt, HERE / "best_model.pt")
print(f"\nbest LR = {best_lr:.0e} (val={results[best_lr][0]:.6f}) -> best_model.pt に反映")
