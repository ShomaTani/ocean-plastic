"""
沿岸放流地点の自動生成

使い方:
  python gen_release_sites.py <ncファイル> --n_sites 300 --out data/release_sites.csv
"""
import argparse
import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, label

p = argparse.ArgumentParser()
p.add_argument("nc_path")
p.add_argument("--var_u", default="uo")
p.add_argument("--n_sites", type=int, default=300)
p.add_argument("--min_gap_deg", type=float, default=0.0,
               help="地点間の最小間隔(度)。0なら間隔制御せず単純間引き")
p.add_argument("--out", default="release_sites.csv")
p.add_argument("--plot", default="release_sites_preview.png")
args = p.parse_args()

ds = xr.open_dataset(args.nc_path)
u0 = ds[args.var_u]
if "time" in u0.dims:
    u0 = u0.isel(time=0)
if "depth" in u0.dims:
    u0 = u0.isel(depth=0)
lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values

land = np.isnan(u0.values)
print(f"grid: {land.shape}, land fraction: {land.mean():.1%}")

# 沿岸の海セル = 陸に4近傍で隣接する海セル
land_dilated = binary_dilation(land, iterations=1)
coastal_ocean = land_dilated & (~land)
ys, xs = np.where(coastal_ocean)
print(f"coastal ocean cells: {len(ys)}")

if len(ys) == 0:
    raise RuntimeError("沿岸セルが見つからない。land mask の抽出方法を確認して")

coords = np.column_stack([lon[xs], lat[ys]])

if args.min_gap_deg > 0:
    # 貪欲法: 既に選んだ点から min_gap_deg 未満の候補を除外しながら選ぶ
    rng = np.random.default_rng(0)
    order = rng.permutation(len(coords))
    chosen = []
    chosen_arr = np.empty((0, 2))
    for idx in order:
        c = coords[idx]
        if len(chosen_arr) == 0 or np.min(np.hypot(*(chosen_arr - c).T)) >= args.min_gap_deg:
            chosen.append(c)
            chosen_arr = np.array(chosen)
        if len(chosen) >= args.n_sites:
            break
    sites = np.array(chosen)
else:
    # 単純な等間隔インデックス間引き(海岸線に沿った並び順は保証しないが手軽)
    idx = np.linspace(0, len(coords) - 1, args.n_sites).astype(int)
    sites = coords[idx]

print(f"selected {len(sites)} release sites (requested {args.n_sites})")

import pandas as pd
df = pd.DataFrame(sites, columns=["lon", "lat"])
df.insert(0, "site_id", range(len(df)))
df.to_csv(args.out, index=False)
print(f"saved: {args.out}")

# プレビュー画像
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    land_rgba = np.zeros((*land.shape, 4))
    land_rgba[land] = [0.85, 0.85, 0.85, 1.0]
    ax.imshow(land_rgba, origin="lower",
              extent=[lon.min(), lon.max(), lat.min(), lat.max()], aspect="auto")
    ax.scatter(sites[:, 0], sites[:, 1], s=8, c="red", zorder=3)
    ax.set_title(f"{len(sites)} auto-generated coastal release sites")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.tight_layout()
    plt.savefig(args.plot, dpi=130)
    print(f"saved: {args.plot}")
except Exception as e:
    print("plot skipped:", e)