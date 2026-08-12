"""
TRACE — bexp04: 逆モデル用のweight decay再探索

forward/sweep_wd.pyと同じやり方。LRはbexp03で確定(3e-3のまま)。
"""

import shutil
from pathlib import Path

import train

CANDIDATES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
HERE = Path(__file__).resolve().parent

results = {}
for wd in CANDIDATES:
    train.WEIGHT_DECAY = wd
    train.CKPT_PATH = HERE / f"sweep_wd_{wd:.0e}.pt"
    print(f"\n===== weight_decay={wd:.0e} =====")
    best_val, train_at_best = train.main()
    results[wd] = (best_val, train_at_best)

print("\n===== summary =====")
for wd, (best_val, train_at_best) in sorted(results.items(), key=lambda kv: kv[1][0]):
    gap = best_val - train_at_best
    print(f"wd={wd:.0e}  best_val={best_val:.6f}  train@best={train_at_best:.6f}  gap={gap:.6f}")

best_wd = min(results, key=lambda k: results[k][0])
best_ckpt = HERE / f"sweep_wd_{best_wd:.0e}.pt"
shutil.copy(best_ckpt, HERE / "best_model.pt")
print(f"\nbest weight_decay = {best_wd:.0e} (val={results[best_wd][0]:.6f}) -> best_model.pt に反映")
