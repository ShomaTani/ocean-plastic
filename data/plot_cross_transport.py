"""
越境輸送(日本海横断など)だけを抜き出して可視化

使い方:
  python plot_cross_transport.py <GLORYSのncファイル> [--zarr trace_run.zarr]
                                 [--from Korea] [--max_lines 400]
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("nc_path", help="陸地マスク描画用のGLORYSファイル")
p.add_argument("--zarr", default="trace_run.zarr")
p.add_argument("--var_u", default="uo")
p.add_argument("--from_region", default=None,
               help="この地域から出た粒子だけ描画 (Korea/China/Japan)。省略時は全越境")
p.add_argument("--max_lines", type=int, default=400,
               help="描画する軌跡の最大本数(多すぎると潰れる)")
p.add_argument("--out", default=None)
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

start_reg = np.array([region_of(a, b) for a, b in zip(slon, slat)])
end_reg = np.array([region_of(a, b) for a, b in zip(flon, flat)])

valid = (beached == 1) & (left_domain == 0)
cross = valid & (start_reg != end_reg)
if args.from_region:
    cross = cross & (start_reg == args.from_region)

idx = np.where(cross)[0]
print(f"越境輸送した粒子: {len(idx)}")
if len(idx) > args.max_lines:
    idx = np.random.default_rng(0).choice(idx, args.max_lines, replace=False)
    print(f"  → 見やすさのため {args.max_lines} 本をランダム抽出して描画")

# ---- 陸地マスク ----
g = xr.open_dataset(args.nc_path)
u0 = g[args.var_u]
if "time" in u0.dims:
    u0 = u0.isel(time=0)
if "depth" in u0.dims:
    u0 = u0.isel(depth=0)
lon = g["longitude"].values if "longitude" in g.coords else g["lon"].values
lat = g["latitude"].values if "latitude" in g.coords else g["lat"].values
land = np.isnan(u0.values)
if lat[0] > lat[-1]:
    lat = lat[::-1]; land = land[::-1, :]
lon_min, lon_max = float(lon.min()), float(lon.max())
lat_min, lat_max = float(lat.min()), float(lat.max())

fig, ax = plt.subplots(figsize=(10, 8))
land_rgba = np.zeros((*land.shape, 4))
land_rgba[land] = [0.82, 0.82, 0.82, 1.0]
land_rgba[~land] = [0, 0, 0, 0]
ax.imshow(land_rgba, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max],
          aspect="auto", zorder=1)
ax.set_facecolor("#dceefb")

# 起点地域ごとに色分け
colors = {"Korea": "#d62728", "China": "#ff7f0e",
          "China_NE/Russia": "#9467bd", "Japan": "#1f77b4", "Other": "#7f7f7f"}
drawn = set()
for i in idx:
    c = colors.get(start_reg[i], "#7f7f7f")
    lbl = start_reg[i] if start_reg[i] not in drawn else None
    drawn.add(start_reg[i])
    ax.plot([slon[i], flon[i]], [slat[i], flat[i]],
            c=c, lw=0.5, alpha=0.35, zorder=2, label=lbl)
    ax.scatter(flon[i], flat[i], s=4, c=c, alpha=0.6, zorder=3)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
title = "Cross-border transport (start → beaching)"
if args.from_region:
    title += f" from {args.from_region}"
ax.set_title(title)
ax.legend(loc="upper right", fontsize=8, title="origin region")
plt.tight_layout()
out = args.out or f"cross_transport{'_' + args.from_region if args.from_region else ''}.png"
plt.savefig(out, dpi=150)
print(f"saved: {out}")