
"""
TRACE — 小規模疎通テスト (10地点 × 10粒子 × 30日)
素の OceanParcels (parcels 3.x) で CMEMS GLORYS12V1 を直読み。

目的: 「粒子放流 → 移流 + beaching → .zarr 出力 → 密度マップ変換」の
      全工程を一度末端まで流し、本番バッチ(④)前に落とし穴を潰す。

このテストが通れば、④は RELEASE_SITES を ~300 点に、N_PER_SITE と
RUNTIME_DAYS を増やすだけでそのままスケールする。

依存: pip install parcels xarray zarr netcdf4   (matplotlib は任意)
"""

import math
import numpy as np
import xarray as xr
from datetime import timedelta
from parcels import (FieldSet, ParticleSet, JITParticle, Variable,
                     StatusCode, Field)

# =====================================================================
# CONFIG — ここだけ自分の環境に合わせて編集
# =====================================================================
GLORYS_FILE = "raw/glorys_2018_2022_surface_uovo.nc"   # DL済み GLORYS12V1 NetCDF のパス
VAR_U, VAR_V = "uo", "vo"              # 表層東西/南北流速の変数名 (GLORYS 既定)
DIM_LON, DIM_LAT = "longitude", "latitude"   # 座標名 (GLORYS 既定)
DIM_TIME, DIM_DEPTH = "time", "depth"        # depth が無ければ None にする

RUNTIME_DAYS = 90      # 追跡日数 (本番: 90日)
DT_MIN       = 30      # 移流タイムステップ(分)
OUTPUT_DT_H  = 24      # 軌跡を何時間ごとに保存するか
N_PER_SITE   = 50      # 1放流地点あたりの粒子数 (本番: 50)
JITTER_DEG   = 0.05    # 各地点まわりに粒子を散らす幅(度)
GRID_N       = 128     # 密度マップの解像度 (最終的な NN 入出力に合わせる)
OUT_ZARR     = "trace_run.zarr"

# 放流地点: gen_release_sites.py が出力した CSV から読み込む
# (陸マス上の点は自動生成の時点で既に除外済みなので、ここでの
#  陸判定スキップはほぼ発生しないはず)
RELEASE_SITES_CSV = "release_sites.csv"   # gen_release_sites.py の --out と合わせる

import pandas as pd
_sites_df = pd.read_csv(RELEASE_SITES_CSV)
RELEASE_SITES = list(zip(_sites_df["lon"], _sites_df["lat"]))
print(f"    loaded {len(RELEASE_SITES)} release sites from {RELEASE_SITES_CSV}")

# =====================================================================
# 1. GLORYS を開いて座標・land mask・表層深度を取得
# =====================================================================
print("[1] loading GLORYS ...")
ds0 = xr.open_dataset(GLORYS_FILE)
lons = ds0[DIM_LON].values
lats = ds0[DIM_LAT].values

u0 = ds0[VAR_U].isel({DIM_TIME: 0})
if DIM_DEPTH and DIM_DEPTH in u0.dims:
    u0 = u0.isel({DIM_DEPTH: 0})
land = np.isnan(u0.values).astype(np.float32)   # 1=陸(欠損), 0=海
depth0 = float(ds0[DIM_DEPTH].values[0]) if (DIM_DEPTH and DIM_DEPTH in ds0.dims) else 0.0
print(f"    grid: {lons.size} x {lats.size}, ocean cells: {(land < 0.5).sum()}, "
      f"release depth: {depth0:.2f} m")

# lat が降順(N→S)だと parcels が U/V を反転させる場合がある。
# その場合は land mask もずれるので、ここで昇順に揃えておく。
if lats[0] > lats[-1]:
    print("    ! latitude is descending — flipping mask to match parcels")
    lats = lats[::-1]
    land = land[::-1, :]

# =====================================================================
# 2. FieldSet 構築 + land mask を単位変換なしの別フィールドとして追加
# =====================================================================
print("[2] building fieldset ...")
filenames  = {"U": GLORYS_FILE, "V": GLORYS_FILE}
variables  = {"U": VAR_U, "V": VAR_V}
dimensions = {"lon": DIM_LON, "lat": DIM_LAT, "time": DIM_TIME}
indices = None
if DIM_DEPTH and DIM_DEPTH in ds0.dims:
    dimensions["depth"] = DIM_DEPTH
    indices = {"depth": [0]}   # 表層のみ読み込み(2D扱い)

fieldset = FieldSet.from_netcdf(filenames, variables, dimensions,
                                indices=indices, allow_time_extrapolation=False)
# mesh='flat' + nearest で 0/1 をそのまま返す(spherical だと deg/s 変換が入って壊れる)
lmask = Field("landmask", land, lon=lons, lat=lats, mesh="flat",
              interp_method="nearest")
fieldset.add_field(lmask)

# =====================================================================
# 3. 放流地点を海セルに限定し、ジッタを付けて粒子配列を生成
# =====================================================================
def nearest(arr, v):
    return int(np.abs(arr - v).argmin())

rng = np.random.default_rng(0)
plon, plat, porigin = [], [], []
n_resampled_total = 0
for k, (lo, la) in enumerate(RELEASE_SITES):
    j, i = nearest(lats, la), nearest(lons, lo)
    if land[j, i] > 0.5:
        print(f"    ! site {k} ({lo},{la}) is on land — skipped")
        continue
    # ジッタ後の座標が陸に落ちていないか1粒子ずつ検証する。
    # (地点の中心座標だけ陸判定しても、ジッタ(JITTER_DEG)がグリッド1マス分
    #  近いため、ジッタ後に隣の陸セルへ飛んでしまうケースがあるため)
    n_ok = 0
    n_try = 0
    max_try = N_PER_SITE * 20  # 無限ループ防止
    while n_ok < N_PER_SITE and n_try < max_try:
        jlo = lo + rng.uniform(-JITTER_DEG, JITTER_DEG)
        jla = la + rng.uniform(-JITTER_DEG, JITTER_DEG)
        jj, ji = nearest(lats, jla), nearest(lons, jlo)
        n_try += 1
        if land[jj, ji] > 0.5:
            n_resampled_total += 1
            continue  # 陸に落ちた → 破棄して再サンプリング
        plon.append(jlo); plat.append(jla); porigin.append(k)
        n_ok += 1
    if n_ok < N_PER_SITE:
        print(f"    ! site {k}: {N_PER_SITE - n_ok}粒子が{max_try}回試行後も海に置けず断念"
              f"(狭い入り江等の可能性)")
plon = np.array(plon); plat = np.array(plat); porigin = np.array(porigin, np.int32)
print(f"[3] releasing {plon.size} particles from {len(set(porigin))} ocean sites")
if n_resampled_total > 0:
    print(f"    (ジッタが陸に落ちて再サンプリングした回数: {n_resampled_total} "
          f"— これが多いほど、修正前は誤beachingが多かったはず)")

# =====================================================================
# 4. 粒子クラス + カーネル (合成フィールドで検証済み)
# =====================================================================
class Plastic(JITParticle):
    beached     = Variable("beached",     dtype=np.int32,   initial=0)  # 陸マスクに接触=本当の漂着
    left_domain = Variable("left_domain", dtype=np.int32,   initial=0)  # 領域外に出て追跡不能
    prev_lon    = Variable("prev_lon",    dtype=np.float32, initial=0.)
    prev_lat    = Variable("prev_lat",    dtype=np.float32, initial=0.)
    origin      = Variable("origin",      dtype=np.int32,   initial=0, to_write="once")

def StorePrev(particle, fieldset, time):
    if particle.beached == 0 and particle.left_domain == 0:
        particle.prev_lon = particle.lon
        particle.prev_lat = particle.lat

def AdvectBeach(particle, fieldset, time):          # 漂着済み/領域外は動かさない RK4
    if particle.beached == 0 and particle.left_domain == 0:
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon, particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5*particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5*particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle_dlon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle_dlat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

def BeachOnLand(particle, fieldset, time):          # 陸セルに入ったら直前位置で漂着(本物の漂着)
    if particle.beached == 0 and particle.left_domain == 0:
        if fieldset.landmask[time, particle.depth, particle.lat, particle.lon] > 0.5:
            particle.lon = particle.prev_lon
            particle.lat = particle.prev_lat
            particle.beached = 1

def Recover(particle, fieldset, time):
    # 領域外に出た/補間エラー = 「追跡不能」であって「漂着」ではないので
    # beached とは別のフラグ(left_domain)に立てる。密度マップの集計から除外するため。
    if (particle.state == StatusCode.ErrorOutOfBounds or
            particle.state == StatusCode.ErrorInterpolation):
        particle.lon = particle.prev_lon
        particle.lat = particle.prev_lat
        particle.left_domain = 1
        particle.state = StatusCode.Success

# =====================================================================
# 5. 実行
# =====================================================================
print("[5] running simulation ...")
pset = ParticleSet(fieldset, pclass=Plastic,
                   lon=plon, lat=plat,
                   depth=np.full(plon.size, depth0),
                   origin=porigin)
pfile = pset.ParticleFile(OUT_ZARR, outputdt=timedelta(hours=OUTPUT_DT_H))
pset.execute([StorePrev, AdvectBeach, BeachOnLand, Recover],
             runtime=timedelta(days=RUNTIME_DAYS),
             dt=timedelta(minutes=DT_MIN),
             output_file=pfile)

# =====================================================================
# 6. 後処理: 軌跡 → origin ごとの漂着密度マップ (学習データ本体)
#    集計対象は2種類を両方保存して比較できるようにする:
#      - all   : 全粒子(浮遊中も含む)の最終位置
#      - beach : 漂着済み(beached==1)の粒子のみ
# =====================================================================
print("[6] building per-origin density maps (all & beached-only) ...")
ds = xr.open_zarr(OUT_ZARR)
flon = ds.lon.isel(obs=-1).values
flat = ds.lat.isel(obs=-1).values
beached     = ds.beached.isel(obs=-1).values
left_domain = ds.left_domain.isel(obs=-1).values
origin      = ds.origin.values

# 領域外に出た粒子(left_domain==1)は「境界で凍結された位置」であって物理的な
# 意味を持たないため、all/beach 両方の密度マップから除外する。
in_domain = (left_domain == 0)

gx = np.linspace(float(lons.min()), float(lons.max()), GRID_N + 1)
gy = np.linspace(float(lats.min()), float(lats.max()), GRID_N + 1)

n_sites = len(RELEASE_SITES)
maps_all   = np.zeros((n_sites, GRID_N, GRID_N), dtype=np.float32)
maps_beach = np.zeros((n_sites, GRID_N, GRID_N), dtype=np.float32)
beach_frac_by_site      = np.zeros(n_sites, dtype=np.float32)
left_domain_frac_by_site = np.zeros(n_sites, dtype=np.float32)
n_particles_by_site = np.zeros(n_sites, dtype=np.int32)

for k in range(n_sites):
    sel = (origin == k)
    n_k = sel.sum()
    n_particles_by_site[k] = n_k
    if n_k == 0:
        continue  # 陸判定でスキップされた地点(release時に弾かれた)。ゼロ埋めのまま

    left_domain_frac_by_site[k] = (sel & ~in_domain).sum() / n_k

    sel_in = sel & in_domain   # 領域内に留まった粒子のみを対象にする
    if sel_in.sum() == 0:
        continue  # 全粒子が境界離脱。このoriginは有効なマップを作れない

    Ha, _, _ = np.histogram2d(flon[sel_in], flat[sel_in], bins=[gx, gy])
    Ha = Ha.T
    if Ha.sum() > 0:
        Ha = Ha / Ha.sum()
    maps_all[k] = Ha

    sel_b = sel_in & (beached == 1)
    n_in = sel_in.sum()
    beach_frac_by_site[k] = sel_b.sum() / n_in if n_in > 0 else 0.0
    if sel_b.sum() > 0:
        Hb, _, _ = np.histogram2d(flon[sel_b], flat[sel_b], bins=[gx, gy])
        Hb = Hb.T
        if Hb.sum() > 0:
            Hb = Hb / Hb.sum()
        # Hb.sum()==0 は beached==1 なのに座標がグリッド範囲外(境界丸め等)の
        # レアケース。ゼロ配列のまま残し、警告なく次のoriginへ進む。
        maps_beach[k] = Hb
    # sel_b が0件(90日でも漂着なし)の origin はゼロ配列のまま
    # → 学習時にこの origin をどう扱うかは別途要検討(サンプル除外 or 明示フラグ)

np.save("density_maps_all.npy", maps_all)       # shape: (n_sites, GRID_N, GRID_N)
np.save("density_maps_beach.npy", maps_beach)   # shape: (n_sites, GRID_N, GRID_N)
_sites_df.to_csv("release_sites_used.csv", index=False)  # origin index との対応表
print(f"    saved density_maps_all.npy / density_maps_beach.npy  "
      f"shape={maps_all.shape}")
print(f"    mean beached fraction (領域内粒子中)     : {beach_frac_by_site.mean():.2f}")
print(f"    mean left_domain fraction (領域外離脱)   : {left_domain_frac_by_site.mean():.2f}")
print(f"    sites with zero beached particles : {(beach_frac_by_site == 0).sum()} / {n_sites}")
if left_domain_frac_by_site.mean() > 0.3:
    print("  !! 平均3割超が境界離脱。ドメインが狭すぎる可能性 → 領域を広げるか、"
          "境界離脱を許容前提の設計にするか要検討")

empty_origins = np.where(n_particles_by_site == 0)[0]
if len(empty_origins) > 0:
    print(f"    !! 粒子ゼロのorigin(陸判定でskip等) {len(empty_origins)}件: "
          f"{list(empty_origins[:10])}{' ...' if len(empty_origins) > 10 else ''}")

# 目視用のorigin: origin 0 が空なら、実際に粒子＆漂着があるものを自動選択する
# (origin 0 固定だと land-skip 発生時にグラフが常に真っ暗になり気づきにくいため)
candidates = np.where((n_particles_by_site > 0) & (beach_frac_by_site > 0))[0]
if len(candidates) > 0:
    plot_origin = int(candidates[0])
elif np.any(n_particles_by_site > 0):
    plot_origin = int(np.where(n_particles_by_site > 0)[0][0])
    print(f"    (origin {plot_origin} は粒子はあるが漂着ゼロ。all側のみ意味を持つ)")
else:
    plot_origin = None
    print("    !! 全origin粒子ゼロ。放流点が全滅している可能性大 — 上流の land mask 判定を要確認")

H = maps_all[plot_origin] if plot_origin is not None else np.zeros((GRID_N, GRID_N))

# =====================================================================
# 7. 合否チェック — ここが全部 OK なら本番バッチへ進んでよい
# =====================================================================
start_lon = ds.lon.isel(obs=0).values
disp = np.hypot(flon - start_lon, flat - ds.lat.isel(obs=0).values)
print("\n===== PASS CHECK =====")
print(f"particles           : {plon.size}")
print(f"beached fraction(全体) : {beached.mean():.2f}   (0や1に張り付いてたら要注意)")
print(f"left_domain fraction(全体) : {left_domain.mean():.2f}   (境界離脱率。高すぎたら領域を疑う)")
print(f"mean displacement    : {np.nanmean(disp):.3f} deg  (>0 でないと移流できてない)")
print(f"max  displacement    : {np.nanmax(disp):.3f} deg")
if plot_origin is not None:
    print(f"density map sum(origin={plot_origin},all)   : {maps_all[plot_origin].sum():.3f}   (1.0 なら正規化OK)")
    print(f"density map sum(origin={plot_origin},beach) : {maps_beach[plot_origin].sum():.3f}   (漂着0件なら0.0のまま)")
if np.nanmean(disp) < 1e-3:
    print("  !! ほぼ動いてない → 変数名/放流点が陸/時間範囲を疑う")
if beached.mean() > 0.99:
    print("  !! 即死の可能性 → land mask の向き(lat昇降)を疑う")
if (beach_frac_by_site == 0).sum() > n_sites * 0.3:
    print("  !! 3割超のoriginで漂着ゼロ → RUNTIME_DAYSを延ばすか、beaching判定を確認")
print("======================")

# 任意: 実際に粒子のあるoriginでの all/beach 比較プロット
try:
    if plot_origin is None:
        raise RuntimeError("有効なoriginが無いためプロットをスキップ")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, Hshow, title in zip(
            axes, [maps_all[plot_origin], maps_beach[plot_origin]],
            ["all particles (incl. drifting)", "beached only"]):
        mesh = ax.pcolormesh(gx[:-1], gy[:-1], Hshow, shading="auto", vmin=0)
        ax.scatter(RELEASE_SITES[plot_origin][0], RELEASE_SITES[plot_origin][1],
                   c="r", marker="*", s=120, edgecolors="white", linewidths=0.8)
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    plt.suptitle(f"origin {plot_origin} (n_particles={n_particles_by_site[plot_origin]}): "
                f"all vs. beached-only density")
    plt.tight_layout()
    plt.savefig("density_origin_check.png", dpi=110)
    print(f"saved density_origin_check.png (origin={plot_origin})")
except Exception as e:
    print("plot skipped:", e)