"""
TRACE — 海流場チャンネルの生成(論点A 選択肢3)

GLORYSの生データ(361×301, 高解像度)から時間平均 uo, vo を計算し、
density_maps_beach.npy / train_pairs.npz と同じ128×128グリッドに平均プーリングして
current_field*.npz に保存する。全originで共通の1枚のフィールド(site非依存)。

デフォルト(引数無し)は2018-2022年全体の時間平均(静的、季節非依存)を
current_field.npz に保存する。従来の挙動そのまま。

引数で放出日(YYYY-MM-DD)を渡すと、sim_main.pyの季節データ拡張(複数年の同一
暦日から放出)に合わせて、その日から90日間(=RUNTIME_DAYS)だけの平均を
current_field_{date}.npz に保存する。正規化の平均・標準偏差は5年全体版
(current_field.npz)のものに固定して使う — 各年の冬の海流場を同じスケールで
比較できるようにするため(年ごとに別々の平均・分散で標準化すると、物理的な
差ではなく正規化のズレが混ざってしまう)。

使い方:
  python build_current_field.py                 # 従来通り: 5年全体平均
  python build_current_field.py 2019-01-01       # その90日間だけの平均(季節拡張用)
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d

GLORYS_FILE = "raw/glorys_2018_2022_surface_uovo.nc"
DIM_LON, DIM_LAT = "longitude", "latitude"
VAR_U, VAR_V = "uo", "vo"
GRID_N = 128
RUNTIME_DAYS = 90  # sim_main.pyのRUNTIME_DAYSと合わせる
BASELINE_FILE = "current_field.npz"

RELEASE_DATE = sys.argv[1] if len(sys.argv) > 1 else None
OUT_FILE = f"current_field_{RELEASE_DATE}.npz" if RELEASE_DATE else BASELINE_FILE

# =====================================================================
# 1. GLORYSを開いて時間平均を取る(depth次元は1層しかないのでsqueeze)
#    RELEASE_DATE指定時は、その日から90日間だけに絞って平均する
# =====================================================================
ds = xr.open_dataset(GLORYS_FILE, chunks={"time": 200})
if RELEASE_DATE:
    t0 = np.datetime64(RELEASE_DATE)
    t1 = t0 + np.timedelta64(RUNTIME_DAYS, "D")
    ds = ds.sel(time=slice(t0, t1))
    print(f"[1] loading GLORYS and computing {RELEASE_DATE} 〜 +{RUNTIME_DAYS}日 の time-mean uo, vo "
          f"({ds.sizes['time']}日分) ...")
else:
    print("[1] loading GLORYS and computing 2018-2022 time-mean uo, vo (season-blind baseline) ...")
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
#    RELEASE_DATE指定時は、5年全体版(current_field.npz)の平均・標準偏差を
#    そのまま使う(年ごとに別基準で正規化すると物理的な差と正規化のズレが
#    区別できなくなるため)。無ければ先に baseline を作るよう促して終了する。
# =====================================================================
if RELEASE_DATE:
    if not Path(BASELINE_FILE).exists():
        raise SystemExit(
            f"{BASELINE_FILE} が無い。先に `python build_current_field.py`(引数無し)で"
            f"5年全体のbaselineを作ってから、日付指定を実行すること。"
        )
    baseline = np.load(BASELINE_FILE)
    u_mean_val, u_std_val = float(baseline["u_mean"]), float(baseline["u_std"])
    v_mean_val, v_std_val = float(baseline["v_mean"]), float(baseline["v_std"])
    speed_mean_val, speed_std_val = float(baseline["speed_mean"]), float(baseline["speed_std"])
    print(f"    正規化基準は{BASELINE_FILE}のものを再利用(u_mean={u_mean_val:.4f}, u_std={u_std_val:.4f})")
else:
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
