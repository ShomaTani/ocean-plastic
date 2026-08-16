"""
大陸側(Korea/China/China_NE-Russia)と日本側との行き来(相互輸送)のみを
抜き出して可視化・集計する。

check_cross_transport.py / plot_cross_transport.py の region_of() をそのまま使い、
「大陸→日本」「日本→大陸」の2方向に限定した粒子だけを線で描画し、
表にもまとめる。500地点版データ(trace_run.zarr)に対して実行する想定。

使い方:
  python analyze_continent_japan_transport.py [--zarr trace_run.zarr]
                                               [--nc raw/glorys_2018_2022_surface_uovo.nc]
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
p.add_argument("--nc", default="raw/glorys_2018_2022_surface_uovo.nc")
p.add_argument("--var_u", default="uo")
p.add_argument("--max_lines", type=int, default=2000)
p.add_argument("--out", default="continent_japan_transport.png")
p.add_argument("--out_csv", default="continent_japan_transport_table.csv")
args = p.parse_args()

CONTINENT = {"Korea", "China", "China_NE/Russia"}


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
origin = ds.origin.values

start_reg = np.array([region_of(a, b) for a, b in zip(slon, slat)])
end_reg = np.array([region_of(a, b) for a, b in zip(flon, flat)])

valid = (beached == 1) & (left_domain == 0)
n_total = len(origin)
n_valid = int(valid.sum())

cont_to_jp = valid & np.isin(start_reg, list(CONTINENT)) & (end_reg == "Japan")
jp_to_cont = valid & (start_reg == "Japan") & np.isin(end_reg, list(CONTINENT))
interact = cont_to_jp | jp_to_cont

print(f"総粒子数: {n_total}")
print(f"うち漂着(境界離脱を除く): {n_valid}")
print(f"大陸→日本: {cont_to_jp.sum()}")
print(f"日本→大陸: {jp_to_cont.sum()}")
print(f"大陸-日本間の行き来 合計: {interact.sum()} "
      f"(全粒子の{interact.sum()/n_total*100:.2f}%, 漂着粒子の{interact.sum()/n_valid*100:.2f}%)")

# ---- 表: 起源地域別の内訳 ----
n_from_japan = int(((start_reg == "Japan") & valid).sum())
rows = []
for reg in ["Korea", "China", "China_NE/Russia"]:
    c2j = int(((start_reg == reg) & (end_reg == "Japan") & valid).sum())
    j2c = int(((start_reg == "Japan") & (end_reg == reg) & valid).sum())
    n_from_reg = int(((start_reg == reg) & valid).sum())
    rows.append({
        "地域": reg,
        "地域起源の漂着粒子数": n_from_reg,
        "→Japanに漂着(個)": c2j,
        "→Japan比率(%)": round(c2j / n_from_reg * 100, 2) if n_from_reg else None,
        f"Japan→地域に漂着(個, 分母{n_from_japan})": j2c,
        "Japan→地域比率(%)": round(j2c / n_from_japan * 100, 4) if n_from_japan else None,
    })
table = pd.DataFrame(rows)
print("\n=== 大陸-日本間 行き来テーブル ===")
print(table.to_string(index=False))
table.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
print(f"saved table: {args.out_csv}")

# ---- 可視化 ----
idx = np.where(interact)[0]
if len(idx) > args.max_lines:
    idx = np.random.default_rng(0).choice(idx, args.max_lines, replace=False)

g = xr.open_dataset(args.nc)
u0 = g[args.var_u]
if "time" in u0.dims:
    u0 = u0.isel(time=0)
if "depth" in u0.dims:
    u0 = u0.isel(depth=0)
lon = g["longitude"].values if "longitude" in g.coords else g["lon"].values
lat = g["latitude"].values if "latitude" in g.coords else g["lat"].values
land = np.isnan(u0.values)
if lat[0] > lat[-1]:
    lat = lat[::-1]
    land = land[::-1, :]
lon_min, lon_max = float(lon.min()), float(lon.max())
lat_min, lat_max = float(lat.min()), float(lat.max())

fig, ax = plt.subplots(figsize=(10, 8))
land_rgba = np.zeros((*land.shape, 4))
land_rgba[land] = [0.82, 0.82, 0.82, 1.0]
land_rgba[~land] = [0, 0, 0, 0]
ax.imshow(land_rgba, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max],
          aspect="auto", zorder=1)
ax.set_facecolor("#dceefb")

colors = {"cont_to_jp": "#d62728", "jp_to_cont": "#1f77b4"}
labels = {"cont_to_jp": "Continent -> Japan", "jp_to_cont": "Japan -> Continent"}
drawn = set()
for i in idx:
    key = "cont_to_jp" if cont_to_jp[i] else "jp_to_cont"
    lbl = labels[key] if key not in drawn else None
    drawn.add(key)
    ax.plot([slon[i], flon[i]], [slat[i], flat[i]],
            c=colors[key], lw=0.5, alpha=0.3, zorder=2, label=lbl)
    ax.scatter(flon[i], flat[i], s=4, c=colors[key], alpha=0.6, zorder=3)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("Longitude (deg E)")
ax.set_ylabel("Latitude (deg N)")
ax.set_title(f"Continent <-> Japan transport (n={interact.sum()}, "
             f"{interact.sum()/n_total*100:.2f}% of all particles)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"saved figure: {args.out}")
