"""
全origin分の漂着密度マップを1枚に集約し、陸地マスク(日本列島含む)と
放流地点を重ねて可視化する。「全体としてどこに漂着が集中するか」を見る。

使い方:
  python plot_all.py <GLORYSのncファイル> [--variant beach|all] [--log]
"""
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

p = argparse.ArgumentParser()
p.add_argument("nc_path", help="陸地マスク描画用のGLORYS ncファイル")
p.add_argument("--variant", choices=["beach", "all"], default="beach",
               help="集約対象: beach=漂着済みのみ(推奨) / all=浮遊中も含む")
p.add_argument("--log", action="store_true",
               help="対数スケールで表示(集中と拡散が混在するデータで有効)")
p.add_argument("--var_u", default="uo")
p.add_argument("--out", default=None)
args = p.parse_args()

maps_all   = np.load("density_maps_all.npy")
maps_beach = np.load("density_maps_beach.npy")
sites = pd.read_csv("release_sites_used.csv")
n_sites, GRID_N, _ = maps_all.shape

maps = maps_beach if args.variant == "beach" else maps_all
# 各originのマップは既に合計1に正規化済みなので、単純合計すると
# 「originごとの重み均等」の集約密度になる(粒子数の違いはここでは無視)
agg = maps.sum(axis=0)

# ---- 陸地マスク: GLORYSのNaNパターンから抽出(plot_currents.py と同じ考え方) ----
ds = xr.open_dataset(args.nc_path)
u0 = ds[args.var_u]
if "time" in u0.dims:
    u0 = u0.isel(time=0)
if "depth" in u0.dims:
    u0 = u0.isel(depth=0)
lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
land = np.isnan(u0.values)
if lat[0] > lat[-1]:
    lat = lat[::-1]; land = land[::-1, :]

lon_min, lon_max = float(lon.min()), float(lon.max())
lat_min, lat_max = float(lat.min()), float(lat.max())
gx = np.linspace(lon_min, lon_max, GRID_N + 1)
gy = np.linspace(lat_min, lat_max, GRID_N + 1)

fig, ax = plt.subplots(figsize=(10, 8))

# 陸地を先に薄いグレーで描画
land_rgba = np.zeros((*land.shape, 4))
land_rgba[land] = [0.85, 0.85, 0.85, 1.0]
land_rgba[~land] = [0, 0, 0, 0]
ax.imshow(land_rgba, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max],
          aspect="auto", zorder=1)

# 集約密度ヒートマップ
agg_masked = np.ma.masked_where(agg <= 0, agg)
if args.log and agg_masked.count() > 0:
    norm = LogNorm(vmin=max(agg_masked.min(), 1e-4), vmax=agg_masked.max())
else:
    norm = None
mesh = ax.pcolormesh(gx[:-1], gy[:-1], agg_masked, shading="auto",
                     cmap="inferno", norm=norm, zorder=2, alpha=0.9)
fig.colorbar(mesh, ax=ax, label=f"aggregated density ({args.variant}, "
            f"{'log' if args.log else 'linear'} scale, summed over {n_sites} origins)",
            pad=0.02)

# 放流地点を小さい点で重ねる
ax.scatter(sites["lon"], sites["lat"], s=6, c="cyan", edgecolors="none",
          alpha=0.7, zorder=3, label="release sites")

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(f"Aggregated beaching density across {n_sites} origins "
            f"({args.variant}{', log-scale' if args.log else ''})")
ax.legend(loc="upper right", fontsize=8)
ax.set_facecolor("#dceefb")

plt.tight_layout()
out = args.out or f"aggregate_density_{args.variant}{'_log' if args.log else ''}.png"
plt.savefig(out, dpi=150)
print(f"saved: {out}")