"""
TRACE — 入力データの整形: 放出地点(lon, lat) → ガウシアン入力マップ
論点A(CLAUDE.md §6)の選択肢2を採用。放出点を中心に正規分布で滲ませた
128×128マップを作り、対応する漂着密度マップ(density_maps_beach_{date}.npy)と
ペアにして1ファイルにまとめる。

季節データ拡張(複数年の同一暦日から放出、sim_main.py参照)に対応するため、
RELEASE_DATESに列挙した全ての放出日について site×date の組合せをそれぞれ
1サンプルとして連結する。同じ地点でも放出日が違えば海流場が違うため出力
(漂着分布)が変わりうる — これが学習データに季節性(のうち経年変動)の
シグナルを持たせる部分。

日付を増やしたい場合は、先に sim_main.py <date> と build_current_field.py <date>
を実行して density_maps_beach_{date}.npy / current_field_{date}.npz を用意した上で、
下のRELEASE_DATESに追記して再実行すればよい(既存日付の再計算は不要)。

出力: train_pairs.npz
  X          : (n_sites*n_dates, 128, 128) float32 — 入力(ガウシアン放出マップ、sum=1)
  Y          : (n_sites*n_dates, 128, 128) float32 — 出力(漂着密度マップ、sum=1 or 0)
  site_id, lon, lat : (n_sites*n_dates,)            — release_sites_used.csv 由来(date分だけ複製)
  release_date : (n_sites*n_dates,) str             — このサンプルの放出日("YYYY-MM-DD")
  valid      : (n_sites*n_dates,) bool              — Y.sum() > 0 かどうか(論点C参照)
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

# =====================================================================
# CONFIG
# =====================================================================
GLORYS_FILE = "raw/glorys_2018_2022_surface_uovo.nc"  # sim_main.py と同じグリッド定義を再現するため座標だけ読む
DIM_LON, DIM_LAT = "longitude", "latitude"

RELEASE_SITES_CSV = "release_sites_used.csv"   # sim_main.py が保存した origin index との対応表
# 季節データ拡張: 各年1/1起点の5冬分(sim_main.py <date> で生成済み)
RELEASE_DATES = ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]
GRID_N   = 128
SIGMA_PX = 1.0    # ガウシアンの広がり(ピクセル単位)。1セル ≒ 0.2度前後
OUT_FILE = "train_pairs.npz"

# =====================================================================
# 1. 放出地点とグリッド定義を読み込む(sim_main.py の gx, gy を再現)
# =====================================================================
print("[1] loading release sites and grid definition ...")
sites = pd.read_csv(RELEASE_SITES_CSV)
n_sites = len(sites)

ds0 = xr.open_dataset(GLORYS_FILE)
lons = ds0[DIM_LON].values
lats = ds0[DIM_LAT].values
gx = np.linspace(float(lons.min()), float(lons.max()), GRID_N + 1)
gy = np.linspace(float(lats.min()), float(lats.max()), GRID_N + 1)
print(f"    grid: lon [{gx[0]:.2f}, {gx[-1]:.2f}] x lat [{gy[0]:.2f}, {gy[-1]:.2f}], {GRID_N}x{GRID_N}")
print(f"    release dates: {RELEASE_DATES}")

# =====================================================================
# 2. 各放出地点を (row, col) ピクセルに変換
#    histogram2d(bins=[gx, gy]) と同じ一様ビン規則(floor + clip)で揃える
#    → row=lat方向, col=lon方向 (sim_main.py の Ha.T と同じ向き)
# =====================================================================
def lonlat_to_pixel(lon, lat):
    col = np.clip(np.floor((lon - gx[0]) / (gx[1] - gx[0])).astype(int), 0, GRID_N - 1)
    row = np.clip(np.floor((lat - gy[0]) / (gy[1] - gy[0])).astype(int), 0, GRID_N - 1)
    return row, col

rows, cols = lonlat_to_pixel(sites["lon"].values, sites["lat"].values)

# =====================================================================
# 3. 各地点に one-hot を置いて Gaussian カーネルで畳み込む(ぼかす)
#    放出地点の(lon,lat)は放出日によらず同じなので、Xはsite単位で1回だけ作り、
#    date分だけタイル(複製)する
# =====================================================================
print("[2] building gaussian input maps (one-hot + gaussian_filter) ...")
X_per_site = np.zeros((n_sites, GRID_N, GRID_N), dtype=np.float32)
for k in range(n_sites):
    onehot = np.zeros((GRID_N, GRID_N), dtype=np.float64)
    onehot[rows[k], cols[k]] = 1.0
    g = gaussian_filter(onehot, sigma=SIGMA_PX, mode="constant")
    X_per_site[k] = (g / g.sum()).astype(np.float32)

# =====================================================================
# 4. 放出日ごとに対応する出力(漂着密度マップ)を読み込んでペアにし、連結する
# =====================================================================
print("[3] pairing with output density maps for each release date ...")
X_list, Y_list, site_id_list, lon_list, lat_list, date_list = [], [], [], [], [], []
for date in RELEASE_DATES:
    Y_date = np.load(f"density_maps_beach_{date}.npy")
    assert Y_date.shape == (n_sites, GRID_N, GRID_N), \
        f"shape mismatch: X sites={n_sites}, Y({date})={Y_date.shape}"
    X_list.append(X_per_site)
    Y_list.append(Y_date)
    site_id_list.append(sites["site_id"].values)
    lon_list.append(sites["lon"].values)
    lat_list.append(sites["lat"].values)
    date_list.append(np.full(n_sites, date))
    n_valid = (Y_date.reshape(n_sites, -1).sum(axis=1) > 0).sum()
    print(f"    {date}: {n_valid}/{n_sites} origins have nonzero beach density")

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)
site_id = np.concatenate(site_id_list, axis=0)
lon = np.concatenate(lon_list, axis=0)
lat = np.concatenate(lat_list, axis=0)
release_date = np.concatenate(date_list, axis=0)

valid = Y.reshape(len(Y), -1).sum(axis=1) > 0
print(f"    total: {len(Y)} samples ({n_sites} sites x {len(RELEASE_DATES)} dates), "
      f"{valid.sum()} have nonzero beach density "
      f"(論点C参照、学習時に除外/フラグ扱いを決める)")

# =====================================================================
# 5. 1ファイルにまとめて保存
# =====================================================================
np.savez(OUT_FILE, X=X, Y=Y,
         site_id=site_id, lon=lon, lat=lat,
         release_date=release_date, valid=valid)
print(f"    saved {OUT_FILE}: X{X.shape}, Y{Y.shape}")
