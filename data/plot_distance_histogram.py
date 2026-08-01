"""
漂着までの移動距離を棒グラフ(ヒストグラム)にする。
同地域内での再漂着 と 越境輸送 を色分けして積み上げることで、
「近距離=局所再漂着」「遠距離=越境輸送」の対応を可視化する。

使い方: python plot_distance_histogram.py [--zarr trace_run.zarr]
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--zarr", default="trace_run.zarr")
p.add_argument("--out", default="distance_histogram.png")
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
slon = ds.lon.isel(obs=0).values
slat = ds.lat.isel(obs=0).values
flon = ds.lon.isel(obs=-1).values
flat = ds.lat.isel(obs=-1).values
beached = ds.beached.isel(obs=-1).values
left_domain = ds.left_domain.isel(obs=-1).values

valid = (beached == 1) & (left_domain == 0)
disp_km = np.hypot(flon - slon, flat - slat)[valid] * 111.0  # 概算(1度≈111km)

start_reg = np.array([region_of(a, b) for a, b in zip(slon[valid], slat[valid])])
end_reg = np.array([region_of(a, b) for a, b in zip(flon[valid], flat[valid])])
cross = start_reg != end_reg

# ---- ビン分割 ----
bins = [0, 25, 50, 100, 200, 400, 700, 1000, 10000]
labels = ["0-25", "25-50", "50-100", "100-200", "200-400",
          "400-700", "700-1000", "1000+"]
bin_idx = np.digitize(disp_km, bins) - 1
bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

same_counts = np.array([((bin_idx == i) & ~cross).sum() for i in range(len(labels))])
cross_counts = np.array([((bin_idx == i) & cross).sum() for i in range(len(labels))])
total = same_counts + cross_counts

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(labels))
ax.bar(x, same_counts, label="同地域内 (局所再漂着)", color="#4c72b0")
ax.bar(x, cross_counts, bottom=same_counts, label="越境輸送", color="#dd8452")

for i, t in enumerate(total):
    if t > 0:
        ax.text(i, t + max(total) * 0.01, f"{t}\n({t/valid.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f"{l} km" for l in labels], rotation=30, ha="right")
ax.set_ylabel("粒子数")
ax.set_xlabel("放出地点からの移動距離")
ax.set_title(f"漂着までの移動距離分布 (n={valid.sum()}, "
            f"越境率={cross.sum()/valid.sum()*100:.1f}%)")
ax.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"saved: {args.out}")

print("\n=== ビン別内訳 ===")
for l, s, c in zip(labels, same_counts, cross_counts):
    t = s + c
    print(f"  {l:>10} km: 計{t:5d}件  同地域{s:5d}  越境{c:5d}"
          f"  (越境率{c/t*100:.1f}%)" if t > 0 else f"  {l:>10} km: 0件")