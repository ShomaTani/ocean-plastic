"""
CMEMS GLORYS ダウンロード確認スクリプト
使い方: python check_glorys.py /path/to/your/glorys_file.nc
"""
import sys
import numpy as np
import xarray as xr

if len(sys.argv) < 2:
    print("usage: python check_glorys.py <path_to_nc>")
    sys.exit(1)

path = sys.argv[1]
print(f"opening: {path}\n")
ds = xr.open_dataset(path)

print("===== 変数一覧 =====")
print(list(ds.data_vars))

print("\n===== 座標(次元)一覧 =====")
print(list(ds.coords))
print(dict(ds.sizes))

# 想定される変数名・座標名の候補チェック
u_candidates = ["uo", "u", "eastward_sea_water_velocity"]
v_candidates = ["vo", "v", "northward_sea_water_velocity"]
lon_candidates = ["longitude", "lon"]
lat_candidates = ["latitude", "lat"]

def find(cands, pool):
    return next((c for c in cands if c in pool), None)

vu = find(u_candidates, ds.data_vars)
vv = find(v_candidates, ds.data_vars)
vlon = find(lon_candidates, ds.coords)
vlat = find(lat_candidates, ds.coords)

print(f"\n検出: U変数={vu}, V変数={vv}, lon座標={vlon}, lat座標={vlat}")
if not all([vu, vv, vlon, vlat]):
    print("!! いずれか見つからない変数がある → trace_sim_test.py の VAR_U/VAR_V/DIM_LON/DIM_LAT を手動で合わせて")

print("\n===== 空間範囲 =====")
if vlon and vlat:
    lons = ds[vlon].values
    lats = ds[vlat].values
    print(f"lon: {lons.min():.2f} 〜 {lons.max():.2f}  (東経、東アジアなら120〜150くらいのはず)")
    print(f"lat: {lats.min():.2f} 〜 {lats.max():.2f}  (北緯、日本周辺なら25〜50くらいのはず)")
    print(f"lat の並び: {'昇順(南→北)' if lats[0] < lats[-1] else '降順(北→南)　※要注意、後段で反転処理が必要'}")

print("\n===== 時間範囲 =====")
if "time" in ds.coords:
    t = ds["time"].values
    print(f"開始: {t.min()}")
    print(f"終了: {t.max()}")
    print(f"件数: {t.size}  (日次データなら日数と一致するはず)")

print("\n===== depth次元 =====")
if "depth" in ds.coords:
    d = ds["depth"].values
    print(f"depthあり: {d.size}層、表層={d.min():.2f}m")
else:
    print("depth次元なし(=表層のみのプロダクト。表層のみで問題なし)")

print("\n===== 欠損(陸)パターン確認 =====")
if vu:
    u0 = ds[vu].isel(time=0) if "time" in ds[vu].dims else ds[vu]
    if "depth" in u0.dims:
        u0 = u0.isel(depth=0)
    arr = u0.values
    nan_frac = np.isnan(arr).mean()
    print(f"最初の時刻・表層でのNaN(陸)割合: {nan_frac:.1%}")
    print(f"  → 東アジア広域を切り出していれば、陸地混じりで20〜60%くらいが自然なレンジ")
    if nan_frac < 0.01:
        print("  !! ほぼ陸がない → 切り出し範囲が海のど真ん中だけの可能性、放流点と整合するか確認")
    if nan_frac > 0.95:
        print("  !! ほぼ全部NaN → ダウンロードが壊れている可能性が高い")

print("\n===== 実際の値のレンジ(異常値チェック) =====")
if vu and vv:
    uarr = ds[vu].isel(time=0) if "time" in ds[vu].dims else ds[vu]
    varr = ds[vv].isel(time=0) if "time" in ds[vv].dims else ds[vv]
    if "depth" in uarr.dims:
        uarr = uarr.isel(depth=0); varr = varr.isel(depth=0)
    print(f"U範囲: {np.nanmin(uarr.values):.3f} 〜 {np.nanmax(uarr.values):.3f} m/s")
    print(f"V範囲: {np.nanmin(varr.values):.3f} 〜 {np.nanmax(varr.values):.3f} m/s")
    print("  → 海流の表層速度として、大体 -2〜2 m/s に収まっていれば正常(黒潮域で局所的に3m/s超もあり得る)")

print("\n完了。上の値がおかしければダウンロード or 切り出し範囲を見直して。")