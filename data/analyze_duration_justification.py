"""
90日という追跡期間(RUNTIME_DAYS)の妥当性を検証する。

beached / left_domain は一度立つとその後の全timestepで保持されるフラグなので、
各obs(=日)時点での累積割合を追うと「まだ増え続けているか、頭打ちになっているか」が分かる。
90日時点で新規発生率がほぼゼロまで収束していれば、期間としては妥当と言える。
逆にまだ大きく増加中なら、90日では足りず打ち切りによる過小評価の疑いが残る。

使い方:
  python analyze_duration_justification.py [--zarr trace_run.zarr]
"""
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--zarr", default="trace_run.zarr")
p.add_argument("--out", default="duration_justification.png")
p.add_argument("--out_csv", default="duration_justification_table.csv")
args = p.parse_args()

ds = xr.open_zarr(args.zarr)
beached_all = ds.beached.values          # (n_particles, n_obs)
left_domain_all = ds.left_domain.values  # (n_particles, n_obs)
n_particles, n_obs = beached_all.shape
print(f"粒子数: {n_particles}, obs数(日数): {n_obs}")

day = np.arange(n_obs)  # outputdt=24h => obsインデックス=経過日数
beached_frac = beached_all.mean(axis=0) * 100
left_domain_frac = left_domain_all.mean(axis=0) * 100
floating_frac = 100 - beached_frac - left_domain_frac

# 新規発生率(前日比、パーセンテージポイント/日)
d_beached = np.diff(beached_frac, prepend=beached_frac[0])
d_left = np.diff(left_domain_frac, prepend=left_domain_frac[0])

table = pd.DataFrame({
    "day": day,
    "beached_cum_%": beached_frac.round(3),
    "left_domain_cum_%": left_domain_frac.round(3),
    "floating_%": floating_frac.round(3),
    "new_beached_%pt_per_day": d_beached.round(4),
    "new_left_domain_%pt_per_day": d_left.round(4),
})
table.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

# 主要なマイルストーン
for d in [7, 14, 30, 60, 90]:
    if d - 1 < n_obs:
        i = d - 1
        print(f"day {d:3d}: beached={beached_frac[i]:5.2f}%  left_domain={left_domain_frac[i]:5.2f}%  "
              f"floating={floating_frac[i]:5.2f}%  新規beached率(直近1日)={d_beached[i]:.4f}pt/day  "
              f"新規left_domain率(直近1日)={d_left[i]:.4f}pt/day")

# 最終週(day84-90)の平均新規発生率 vs 最初の週(day0-7)
last_week_beach_rate = d_beached[-7:].mean()
first_week_beach_rate = d_beached[1:8].mean()
last_week_exit_rate = d_left[-7:].mean()
first_week_exit_rate = d_left[1:8].mean()
print(f"\n新規beached率: 最初の週平均={first_week_beach_rate:.4f}pt/day -> "
      f"最終週平均={last_week_beach_rate:.4f}pt/day "
      f"({last_week_beach_rate/first_week_beach_rate*100:.1f}%まで減衰)")
print(f"新規left_domain率: 最初の週平均={first_week_exit_rate:.4f}pt/day -> "
      f"最終週平均={last_week_exit_rate:.4f}pt/day "
      f"({last_week_exit_rate/first_week_exit_rate*100:.1f}%まで減衰)")

print(f"\nday90時点でfloating(未確定)のまま残っている割合: {floating_frac[-1]:.2f}%")
print("  → この割合が「90日で打ち切ったことによる分類未確定」の規模を表す")

# ---- 可視化 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.stackplot(day, beached_frac, floating_frac, left_domain_frac,
             labels=["Beaching (cum.)", "Floating", "Out of range (cum.)"],
             colors=["#2ca02c", "#1f77b4", "#d62728"], alpha=0.85)
ax.axvline(90, color="black", ls="--", lw=1)
ax.set_xlabel("Day")
ax.set_ylabel("Fraction (%)")
ax.set_title("Cumulative fate over time (n=%d)" % n_particles)
ax.set_xlim(0, n_obs - 1)
ax.set_ylim(0, 100)
ax.legend(loc="center right", fontsize=8)

ax = axes[1]
ax.plot(day, d_beached, label="New beaching rate (pt/day)", color="#2ca02c")
ax.plot(day, d_left, label="New out-of-range rate (pt/day)", color="#d62728")
ax.set_xlabel("Day")
ax.set_ylabel("New occurrence rate (percentage-point/day)")
ax.set_title("Daily new-occurrence rate (is it still rising at day 90?)")
ax.set_xlim(0, n_obs - 1)
ax.legend(loc="upper right", fontsize=8)
ax.axhline(0, color="gray", lw=0.5)

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"\nsaved figure: {args.out}")
print(f"saved table: {args.out_csv}")
