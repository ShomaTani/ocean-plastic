"""
TRACE — 入力データの整形: 放出地点(lon, lat) → ガウシアン入力マップ
論点A(CLAUDE.md §6)の選択肢2を採用。放出点を中心に正規分布で滲ませた
128×128マップを作り、対応する漂着密度マップ(density_maps_beach.npy)と
ペアにして1ファイルにまとめる。

出力: train_pairs.npz
  X       : (n_sites, 128, 128) float32 — 入力(ガウシアン放出マップ、sum=1)
  Y       : (n_sites, 128, 128) float32 — 出力(漂着密度マップ、sum=1 or 0)
  site_id, lon, lat : (n_sites,)        — release_sites_used.csv 由来
  valid   : (n_sites,) bool             — Y.sum() > 0 かどうか(論点C参照。
            90日間漂着ゼロのoriginはYがゼロ配列のまま残っている)
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
DENSITY_MAP_NPY   = "density_maps_beach.npy"   # 漂着済みのみの密度マップ(基本こちらを使う)
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
#    → gaussian_filter は端の外側をゼロ扱い(mode="constant")するので、
#      境界付近の地点は g.sum() がわずかに1未満になる。最後に明示的に
#      正規化して、出力側(sum=1)と同じ流儀に揃える
# =====================================================================
print("[2] building gaussian input maps (one-hot + gaussian_filter) ...")
X = np.zeros((n_sites, GRID_N, GRID_N), dtype=np.float32)
for k in range(n_sites):
    onehot = np.zeros((GRID_N, GRID_N), dtype=np.float64)
    onehot[rows[k], cols[k]] = 1.0
    g = gaussian_filter(onehot, sigma=SIGMA_PX, mode="constant")
    X[k] = (g / g.sum()).astype(np.float32)

# =====================================================================
# 4. 対応する出力(漂着密度マップ)を読み込んでペアにする
#    release_sites_used.csv の行順 = origin index = maps 配列のインデックス
# =====================================================================
print("[3] pairing with output density maps ...")
Y = np.load(DENSITY_MAP_NPY)
assert Y.shape == (n_sites, GRID_N, GRID_N), f"shape mismatch: X sites={n_sites}, Y={Y.shape}"

valid = Y.reshape(n_sites, -1).sum(axis=1) > 0
print(f"    {valid.sum()}/{n_sites} origins have nonzero beach density "
      f"({n_sites - valid.sum()} は90日間漂着なし。論点C参照、学習時に除外/フラグ扱いを決める)")

# =====================================================================
# 5. 1ファイルにまとめて保存
# =====================================================================
np.savez(OUT_FILE, X=X, Y=Y,
         site_id=sites["site_id"].values,
         lon=sites["lon"].values, lat=sites["lat"].values,
         valid=valid)
print(f"    saved {OUT_FILE}: X{X.shape}, Y{Y.shape}")
