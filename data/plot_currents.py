"""
GLORYS 表層海流の可視化: 速さ(ヒートマップ) + 向き(矢印)

使い方:
  python plot_currents.py <ncファイル> [--time 0] [--out current_map.png]
"""
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager

# 日本語フォントが使える環境ならそれを使い、無ければ英語ラベルに自動フォールバック
_JP_CANDIDATES = ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "Yu Gothic",
                  "Meiryo", "Noto Sans CJK JP", "IPAexGothic"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_jp_font = next((f for f in _JP_CANDIDATES if f in _available), None)
USE_JP = _jp_font is not None
if USE_JP:
    plt.rcParams["font.family"] = _jp_font
else:
    print("note: 日本語フォントが見つからないため、ラベルを英語で出力します"
          "(Macなら通常 Hiragino Sans が使えるはずです)")

p = argparse.ArgumentParser()
p.add_argument("nc_path")
p.add_argument("--var_u", default="uo")
p.add_argument("--var_v", default="vo")
p.add_argument("--time", type=int, default=0, help="表示する時刻インデックス")
p.add_argument("--out", default="data/current_map.png")
p.add_argument("--quiver_stride", type=int, default=12,
               help="矢印の間引き間隔(値が大きいほど矢印が疎)")
args = p.parse_args()

ds = xr.open_dataset(args.nc_path)
u = ds[args.var_u]
v = ds[args.var_v]
if "time" in u.dims:
    date_label = str(ds["time"].isel(time=args.time).values)[:10]
    u = u.isel(time=args.time)
    v = v.isel(time=args.time)
else:
    date_label = ""
if "depth" in u.dims:
    u = u.isel(depth=0)
    v = v.isel(depth=0)

lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
U = u.values
V = v.values
speed = np.sqrt(U**2 + V**2)

# ---- 陸地レイヤ: NaN(欠損=陸)を薄いグレーで塗る ----
land = np.isnan(U)

fig, ax = plt.subplots(figsize=(9, 7))

# 陸を先に塗る(海より手前に来ないよう最初に描画)
land_rgba = np.zeros((*land.shape, 4))
land_rgba[land] = [0.85, 0.85, 0.85, 1.0]   # 薄いグレー
land_rgba[~land] = [0, 0, 0, 0]             # 海は透明のまま
ax.imshow(land_rgba, origin="lower",
          extent=[lon.min(), lon.max(), lat.min(), lat.max()],
          aspect="auto", zorder=1)

# 流速の大きさをヒートマップ(青系パレット)
cmap = plt.cm.viridis  # 速さは viridis の方が視認性が高い。青統一なら 'Blues' に変更可
mesh = ax.pcolormesh(lon, lat, speed, cmap=cmap, shading="auto",
                     vmin=0, vmax=np.nanpercentile(speed, 98), zorder=2)
cbar = fig.colorbar(mesh, ax=ax,
                    label=("流速の大きさ [m/s]" if USE_JP else "Current speed [m/s]"),
                    pad=0.02)

# 向きを矢印で重ねる(間引いて見やすく)
s = args.quiver_stride
lon2d, lat2d = np.meshgrid(lon, lat)
ax.quiver(lon2d[::s, ::s], lat2d[::s, ::s],
          U[::s, ::s], V[::s, ::s],
          color="white", scale=25, width=0.0025, alpha=0.85, zorder=3)

if USE_JP:
    ax.set_xlabel("経度 (°E)")
    ax.set_ylabel("緯度 (°N)")
    title = "GLORYS12V1 表層海流"
else:
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    title = "GLORYS12V1 Surface Currents"
if date_label:
    title += f" ({date_label})"
ax.set_title(title, fontsize=13)
ax.set_facecolor("#dceefb")  # 海の背景を薄い水色に(NaN以外の描画領域外の余白用)

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"saved: {args.out}")