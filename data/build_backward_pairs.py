"""
TRACE — 逆モデル用データ整形

density_maps_beach_{date}.npy (origin分の漂着密度マップ) を「観測地点」視点に
転置して、逆モデルの学習ペアを作る。新規シミュレーションは不要。

出力: backward_pairs.npz
  X_gaussian   : (n_samples, 128, 128) float32 — 観測地点をガウシアンで滲ませた入力
  Y            : (n_samples, 128, 128) float32 — 責任マップ(sum=1)
  row, col     : (n_samples,) 観測セルのピクセル座標
  lon, lat     : (n_samples,) 観測セルの中心座標(参考用)
  n_origins    : (n_samples,) そのセルに寄与しているorigin数(1なら自明、2以上が本題)
  release_date : (n_samples,) str — このサンプルの放出日("YYYY-MM-DD")
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation, gaussian_filter

GLORYS_FILE = "raw/glorys_2018_2022_surface_uovo.nc"
CURRENT_FIELD_NPZ = "current_field.npz"
DIM_LON, DIM_LAT = "longitude", "latitude"
RELEASE_SITES_CSV = "release_sites_used.csv"
# 季節データ拡張: 各年1/1起点の5冬分(sim_main.py <date> で生成済み)
RELEASE_DATES = ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]
GRID_N = 128
SIGMA_PX = 1.0        # inputdata_dim.pyと同じ値(入力側のガウシアンの広がり)
OUT_FILE = "backward_pairs.npz"

_land = np.load(CURRENT_FIELD_NPZ)["land_mask"]
COASTAL_MASK = binary_dilation(_land) & ~_land

# 1. グリッド定義(sim_main.py / inputdata_dim.pyと同じ)を再現
print("[1] loading grid definition and origin sites ...")
ds0 = xr.open_dataset(GLORYS_FILE)
lons_native = ds0[DIM_LON].values
lats_native = ds0[DIM_LAT].values
gx = np.linspace(float(lons_native.min()), float(lons_native.max()), GRID_N + 1)
gy = np.linspace(float(lats_native.min()), float(lats_native.max()), GRID_N + 1)
cell_lon = (gx[:-1] + gx[1:]) / 2   # 各セルの中心経度(128,)
cell_lat = (gy[:-1] + gy[1:]) / 2   # 各セルの中心緯度(128,)


def lonlat_to_pixel(lon, lat):
    col = np.clip(np.floor((lon - gx[0]) / (gx[1] - gx[0])).astype(int), 0, GRID_N - 1)
    row = np.clip(np.floor((lat - gy[0]) / (gy[1] - gy[0])).astype(int), 0, GRID_N - 1)
    return row, col


sites = pd.read_csv(RELEASE_SITES_CSV)
n_origins = len(sites)
origin_row, origin_col = lonlat_to_pixel(sites["lon"].values, sites["lat"].values)
print(f"    release dates: {RELEASE_DATES}")


def build_pairs_for_date(date):
    """1放出日分の density_maps_beach_{date}.npy を観測地点視点に転置する。"""
    Y_stack = np.load(f"density_maps_beach_{date}.npy")  # (n_origins, 128, 128)
    total = Y_stack.sum(axis=0)          # (128, 128) 全origin合算
    obs_rows, obs_cols = np.where(total > 0)
    n_cells = len(obs_rows)
    print(f"    {date}: {n_cells} 個の観測セル "
          f"(origin1個={((Y_stack > 0).sum(axis=0) == 1).sum()}, "
          f"2個以上={((Y_stack > 0).sum(axis=0) >= 2).sum()})")

    X_gaussian = np.zeros((n_cells, GRID_N, GRID_N), dtype=np.float32)
    Y_resp = np.zeros((n_cells, GRID_N, GRID_N), dtype=np.float32)
    n_contrib = np.zeros(n_cells, dtype=np.int32)
    keep = np.ones(n_cells, dtype=bool)

    for k in range(n_cells):
        r, c = obs_rows[k], obs_cols[k]

        # 入力: 観測セルにone-hot -> gaussian_filterで滲ませる(inputdata_dim.pyと同じ手法)
        onehot = np.zeros((GRID_N, GRID_N), dtype=np.float64)
        onehot[r, c] = 1.0
        g = gaussian_filter(onehot, sigma=SIGMA_PX, mode="constant")
        X_gaussian[k] = (g / g.sum()).astype(np.float32)

        # 正解: そのセルへの各originの寄与を正規化し、origin位置に点質量として積み上げる
        contrib = Y_stack[:, r, c]              # (n_origins,)
        responsibility = contrib / contrib.sum()  # sum=1 (total[r,c]>0が保証済み)
        nz = np.nonzero(contrib)[0]
        n_contrib[k] = len(nz)
        point_mass = np.zeros((GRID_N, GRID_N), dtype=np.float64)
        for i in nz:
            point_mass[origin_row[i], origin_col[i]] += responsibility[i]

        # 点質量のままだとn_origins=1の場合16384マス中1マスだけが非ゼロという
        # ほぼデルタ関数になり、cross_entropyがtrain例の丸暗記を助長してしまう。
        # 入力側と同じgaussian_filterで軽くぼかし、sum=1に再正規化する
        blurred = gaussian_filter(point_mass, sigma=SIGMA_PX, mode="constant")
        # ぼかしは陸/非沿岸にもお構いなしに広がるので、沿岸マスクで切り落として
        # から再正規化する(masked softmaxをかけたモデルの出力空間と揃える)。
        blurred = np.where(COASTAL_MASK, blurred, 0.0)
        total_mass = blurred.sum()
        if total_mass <= 0:
            keep[k] = False
            continue
        Y_resp[k] = (blurred / total_mass).astype(np.float32)

    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"    {date}: 沿岸マスク起因で{n_dropped}セルを除外")
        X_gaussian, Y_resp = X_gaussian[keep], Y_resp[keep]
        obs_rows, obs_cols, n_contrib = obs_rows[keep], obs_cols[keep], n_contrib[keep]

    return X_gaussian, Y_resp, obs_rows, obs_cols, n_contrib


# 2. 放出日ごとに転置し、連結する
print("[2] building input/output pairs for each release date ...")
X_list, Y_list, row_list, col_list, ncontrib_list, date_list = [], [], [], [], [], []
for date in RELEASE_DATES:
    Xg, Yr, r, c, nc = build_pairs_for_date(date)
    X_list.append(Xg)
    Y_list.append(Yr)
    row_list.append(r)
    col_list.append(c)
    ncontrib_list.append(nc)
    date_list.append(np.full(len(r), date))

X_gaussian = np.concatenate(X_list, axis=0)
Y_resp = np.concatenate(Y_list, axis=0)
obs_rows = np.concatenate(row_list, axis=0)
obs_cols = np.concatenate(col_list, axis=0)
n_contrib = np.concatenate(ncontrib_list, axis=0)
release_date = np.concatenate(date_list, axis=0)
print(f"    total: {len(X_gaussian)} samples ({len(RELEASE_DATES)} dates分を連結)")

# 3. 保存
np.savez(
    OUT_FILE,
    X_gaussian=X_gaussian,
    Y=Y_resp,
    row=obs_rows, col=obs_cols,
    lon=cell_lon[obs_cols], lat=cell_lat[obs_rows],
    n_origins=n_contrib,
    release_date=release_date,
)
print(f"    saved {OUT_FILE}: X_gaussian{X_gaussian.shape}, Y{Y_resp.shape}")
