"""
90日後(最終obs)の粒子の運命を3分類する:
  - Beaching  : beached==1 & left_domain==0
  - Floating  : beached==0 & left_domain==0 (領域内をまだ漂流中)
  - Out of range : left_domain==1 (シミュレーション領域(E120-150,N25-50)を離脱)

使い方:
  python analyze_fate_ratios.py [--zarr trace_run.zarr]
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
p.add_argument("--out", default="fate_ratios.png")
p.add_argument("--out_csv", default="fate_ratios_table.csv")
args = p.parse_args()


def region_of(lon, lat):
    if 126.0 <= lon <= 130.0 and 34.0 <= lat <= 43.5:
        return "Korea"
    if lon < 126.0:
        return "China_NE/Russia" if lat >= 40.0 else "China"
    if lon >= 129.0:
        return "Japan"
    return "Other"


ds = xr.open_zarr(args.zarr)
beached = ds.beached.isel(obs=-1).values
left_domain = ds.left_domain.isel(obs=-1).values
slon = ds.lon.isel(obs=0).values
slat = ds.lat.isel(obs=0).values
n_total = len(beached)

fate = np.where(left_domain == 1, "Out of range",
                 np.where(beached == 1, "Beaching", "Floating"))
start_reg = np.array([region_of(a, b) for a, b in zip(slon, slat)])

# ---- 全体の割合 ----
overall = pd.Series(fate).value_counts().reindex(
    ["Beaching", "Floating", "Out of range"]).fillna(0).astype(int)
overall_pct = (overall / n_total * 100).round(2)
print(f"総粒子数: {n_total}\n")
print("=== 全体の内訳 ===")
for k in overall.index:
    print(f"  {k:14s}: {overall[k]:6d} ({overall_pct[k]:5.2f}%)")

# ---- 起源地域別 ----
cross = pd.crosstab(start_reg, fate)
cross = cross.reindex(columns=["Beaching", "Floating", "Out of range"], fill_value=0)
cross_pct = (cross.T / cross.sum(axis=1) * 100).T.round(2)
print("\n=== 起源地域別 内訳(個数) ===")
print(cross)
print("\n=== 起源地域別 内訳(%, 行方向) ===")
print(cross_pct)

# ---- CSV保存 ----
table = cross.copy()
table.columns = [f"{c}(個)" for c in table.columns]
for c in ["Beaching", "Floating", "Out of range"]:
    table[f"{c}(%)"] = cross_pct[c]
table.loc["全体"] = list(overall.values) + list(overall_pct.values)
table.to_csv(args.out_csv, encoding="utf-8-sig")
print(f"\nsaved table: {args.out_csv}")

# ---- 可視化: 積み上げ棒グラフ(全体 + 地域別) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = {"Beaching": "#2ca02c", "Floating": "#1f77b4", "Out of range": "#d62728"}

ax = axes[0]
bottom = 0
for k in ["Beaching", "Floating", "Out of range"]:
    ax.bar(["All particles"], [overall_pct[k]], bottom=bottom, color=colors[k], label=k)
    if overall_pct[k] > 2:
        ax.text(0, bottom + overall_pct[k] / 2, f"{overall_pct[k]:.1f}%",
                ha="center", va="center", fontsize=10, color="white")
    bottom += overall_pct[k]
ax.set_ylabel("Fraction (%)")
ax.set_title("Overall fate at day 90 (n=%d)" % n_total)
ax.set_ylim(0, 100)
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
regions = ["Japan", "Korea", "China", "China_NE/Russia", "Other"]
regions = [r for r in regions if r in cross_pct.index]
bottom = np.zeros(len(regions))
for k in ["Beaching", "Floating", "Out of range"]:
    vals = cross_pct.loc[regions, k].values
    ax.bar(regions, vals, bottom=bottom, color=colors[k], label=k)
    bottom += vals
ax.set_ylabel("Fraction (%)")
ax.set_title("Fate at day 90 by origin region")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=30)
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"saved figure: {args.out}")
