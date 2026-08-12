"""
TRACE — exp05: cross_entropy用のLR再探索

exp04はweighted_mse用に決めたLR=1e-3をそのまま使い回しただけなので、
cross_entropy向けに複数のLRを試して比較する。train.pyのCONFIG(LR以外)は
そのまま固定して使う。checkpointは候補ごとに別名で保存し、最良のものだけ
最後にbest_model.ptとしてコピーする。
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
    best_val = train.main()
    results[lr] = best_val

print("\n===== summary =====")
for lr, val in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"LR={lr:.0e}  best_val={val:.6f}")

best_lr = min(results, key=results.get)
best_ckpt = HERE / f"sweep_lr_{best_lr:.0e}.pt"
shutil.copy(best_ckpt, HERE / "best_model.pt")
print(f"\nbest LR = {best_lr:.0e} (val={results[best_lr]:.6f}) -> best_model.pt に反映")
