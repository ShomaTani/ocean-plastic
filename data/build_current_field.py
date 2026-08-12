"""
TRACE — 海流場チャンネルの生成(論点A 選択肢3)

GLORYSの生データ(361×301, 高解像度)から2018-2022年の時間平均 uo, vo を計算し、
density_maps_beach.npy / train_pairs.npz と同じ128×128グリッドに平均プーリングして
current_field.npz に保存する。全originで共通の1枚のフィールド(site非依存)。

季節変動は無視した静的な平均場(CLAUDE.md §6 論点A参照。放流時期を分けた
シミュレーションをやらない限り、季節別の電流場を学習に使うことはできない)。
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d

GLORYS_FILE = "raw/glorys_2018_2022_surface_uovo.nc"
DIM_LON, DIM_LAT = "longitude", "latitude"
VAR_U, VAR_V = "uo", "vo"
GRID_N = 128
OUT_FILE = "current_field.npz"

# =====================================================================
# 1. GLORYSを開いて時間平均を取る(depth次元は1層しかないのでsqueeze)
# =====================================================================
print("[1] loading GLORYS and computing time-mean uo, vo ...")
ds = xr.open_dataset(GLORYS_FILE, chunks={"time": 200})
u_mean = ds[VAR_U].mean(dim="time").squeeze("depth").compute().values  # (301, 361)
v_mean = ds[VAR_V].mean(dim="time").squeeze("depth").compute().values
lons = ds[DIM_LON].values
lats = ds[DIM_LAT].values
print(f"    u_mean shape={u_mean.shape}, nan比率={np.isnan(u_mean).mean():.2%} (陸マスク)")

# =====================================================================
# 2. sim_main.py / inputdata_dim.py と同じグリッド定義(gx, gy)を再現
# =====================================================================
gx = np.linspace(float(lons.min()), float(lons.max()), GRID_N + 1)
gy = np.linspace(float(lats.min()), float(lats.max()), GRID_N + 1)

# =====================================================================
# 3. 361x301 -> 128x128 に平均プーリングでダウンサンプル
#    NaN(陸)は平均計算から除外する。全部陸のセルはNaNのまま残る
# =====================================================================
print(f"[2] downsampling to {GRID_N}x{GRID_N} via block-mean ...")
lon_grid, lat_grid = np.meshgrid(lons, lats)   # (301, 361) それぞれ

def downsample(field):
    valid = ~np.isnan(field)
    stat, _, _, _ = binned_statistic_2d(
        lon_grid[valid], lat_grid[valid], field[valid],
        statistic="mean", bins=[gx, gy],
    )
    return stat.T   # (lon_bins, lat_bins) -> (lat, lon) に転置してrow=lat, col=lonに揃える

u_128 = downsample(u_mean)
v_128 = downsample(v_mean)
land_mask = np.isnan(u_128)  # 128x128グリッド上での陸マスク(全ビンが陸だったセル)
print(f"    128x128グリッドでの陸セル比率: {land_mask.mean():.2%}")

# =====================================================================
# 4. 流速の大きさ(speed)を物理値(u,v)から先に計算しておく
#    (標準化後のu,vから計算すると単位が混ざっておかしくなるので、この順序が重要)
# =====================================================================
speed_128 = np.sqrt(u_128 ** 2 + v_128 ** 2)  # land_maskのセルはnp.nanのまま伝播する

# =====================================================================
# 5. 標準化(平均0・分散1)して、陸セルは0(=海のセルの平均的な値)で埋める
# =====================================================================
u_mean_val, u_std_val = np.nanmean(u_128), np.nanstd(u_128)
v_mean_val, v_std_val = np.nanmean(v_128), np.nanstd(v_128)
speed_mean_val, speed_std_val = np.nanmean(speed_128), np.nanstd(speed_128)

u_norm = np.where(land_mask, 0.0, (u_128 - u_mean_val) / u_std_val).astype(np.float32)
v_norm = np.where(land_mask, 0.0, (v_128 - v_mean_val) / v_std_val).astype(np.float32)
speed_norm = np.where(land_mask, 0.0, (speed_128 - speed_mean_val) / speed_std_val).astype(np.float32)

np.savez(OUT_FILE, u=u_norm, v=v_norm, speed=speed_norm, land_mask=land_mask,
         u_mean=u_mean_val, u_std=u_std_val, v_mean=v_mean_val, v_std=v_std_val,
         speed_mean=speed_mean_val, speed_std=speed_std_val)
print(f"    saved {OUT_FILE}: u{u_norm.shape}, v{v_norm.shape}, speed{speed_norm.shape}")
