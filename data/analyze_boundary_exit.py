"""
領域外の海流の影響を、既存の軌跡データだけで定量化する。

再シミュレーション無しでできることとして、境界離脱した粒子(left_domain==1)について:
  1. どの境界(東/西/南/北)から抜けたか
  2. 抜ける直前1日間の移動ベクトル(=実効速度。移流+windageの結果)を、
     抜けた境界に対する法線方向(外向き成分)に分解
  3. 何日目に抜けたか
を調べる。

使い方:
  python analyze_boundary_exit.py [--zarr trace_run.zarr]
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
p.add_argument("--sites", default="release_sites_used.csv")
p.add_argument("--out", default="boundary_exit.png")
p.add_argument("--out_csv", default="boundary_exit_table.csv")
args = p.parse_args()

LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 120.0, 150.0, 25.0, 50.0
DEG_KM = 111.0  # 緯度1度の距離(km)。経度方向はcos(lat)で別途補正

ds = xr.open_zarr(args.zarr)
beached = ds.beached.values
left_domain = ds.left_domain.values
lon = ds.lon.values
lat = ds.lat.values
n_particles, n_obs = left_domain.shape

exited = left_domain[:, -1] == 1
idx = np.where(exited)[0]
print(f"境界離脱した粒子: {len(idx)} / {n_particles} ({len(idx)/n_particles*100:.2f}%)")

# 各粒子について、初めてleft_domain==1になったobsインデックスを特定
t_exit = np.argmax(left_domain[idx] == 1, axis=1)  # 最初の1の位置

# 抜けた地点(=その時点のlon/lat。Recoverカーネルによりprev_lonに固定済み)
exit_lon = lon[idx, t_exit]
exit_lat = lat[idx, t_exit]

# 抜ける直前1日間の移動ベクトル(t_exit-1 -> t_exit)。t_exit=0の場合は次のobsとの差で代用
t_prev = np.maximum(t_exit - 1, 0)
t_next = np.minimum(t_exit + 0, n_obs - 1)
dlon = lon[idx, t_exit] - lon[idx, t_prev]
dlat = lat[idx, t_exit] - lat[idx, t_prev]
# t_exit==t_prevのケース(day0離脱)はベクトル計算不能なので除外
valid_vec = t_exit != t_prev

# 度/日 -> km/日 (経度は緯度で補正)
u_kmday = dlon * DEG_KM * np.cos(np.radians(exit_lat))
v_kmday = dlat * DEG_KM

# どの境界から抜けたか(4辺との距離が最小のものを採用)
d_east = np.abs(LON_MAX - exit_lon)
d_west = np.abs(exit_lon - LON_MIN)
d_north = np.abs(LAT_MAX - exit_lat)
d_south = np.abs(exit_lat - LAT_MIN)
d_stack = np.stack([d_east, d_west, d_north, d_south], axis=1)
edge_names = np.array(["East(150E, 黒潮続流側)", "West(120E, 東シナ海側)",
                        "North(50N)", "South(25N)"])
edge = edge_names[np.argmin(d_stack, axis=1)]

# 外向き速度成分(境界に垂直、外向きが正)
outward = np.select(
    [edge == edge_names[0], edge == edge_names[1], edge == edge_names[2], edge == edge_names[3]],
    [u_kmday, -u_kmday, v_kmday, -v_kmday],
)

sites = pd.read_csv(args.sites).set_index("site_id")
origin_id = ds.origin.values[idx]
org_lon = sites.loc[origin_id, "lon"].values
org_lat = sites.loc[origin_id, "lat"].values
o_d_east = np.abs(LON_MAX - org_lon)
o_d_west = np.abs(org_lon - LON_MIN)
o_d_north = np.abs(LAT_MAX - org_lat)
o_d_south = np.abs(org_lat - LAT_MIN)
origin_dist_to_edge = np.min(np.stack([o_d_east, o_d_west, o_d_north, o_d_south], axis=1), axis=1)

df = pd.DataFrame({
    "origin": origin_id,
    "exit_day": t_exit,
    "exit_lon": exit_lon,
    "exit_lat": exit_lat,
    "edge": edge,
    "outward_speed_km_per_day": outward,
    "valid_vec": valid_vec,
    "origin_dist_to_edge_deg": origin_dist_to_edge,
})
# 放出地点自体が境界から1度以内 = ドメイン境界で海岸線が切れているアーティファクト濃厚
df["likely_artifact"] = df["origin_dist_to_edge_deg"] < 1.0
df_v = df[df["valid_vec"]]

print("\n=== 境界別の内訳 ===")
summary = df.groupby("edge").agg(
    n=("edge", "size"),
    pct_of_exits=("edge", lambda s: len(s) / len(df) * 100),
    mean_exit_day=("exit_day", "mean"),
    median_exit_day=("exit_day", "median"),
    mean_origin_dist_to_edge_deg=("origin_dist_to_edge_deg", "mean"),
    pct_likely_artifact=("likely_artifact", lambda s: s.mean() * 100),
).round(2)
summary["pct_of_all_particles"] = (summary["n"] / n_particles * 100).round(3)
print(summary)

n_artifact = int(df["likely_artifact"].sum())
n_genuine = len(df) - n_artifact
print(f"\n=== 境界離脱の内訳: アーティファクト(放出地点が境界1度以内) vs 本物の物理輸送 ===")
print(f"  ドメイン境界アーティファクト濃厚: {n_artifact} 個 "
      f"({n_artifact/len(df)*100:.1f}% of exits, {n_artifact/n_particles*100:.2f}% of all particles)")
print(f"  本物の物理輸送による離脱      : {n_genuine} 個 "
      f"({n_genuine/len(df)*100:.1f}% of exits, {n_genuine/n_particles*100:.2f}% of all particles)")

print("\n=== 境界別の外向き速度(km/日) ===")
speed_summary = df_v.groupby("edge")["outward_speed_km_per_day"].agg(
    ["mean", "median", "std",
     lambda s: (s <= 0).mean() * 100]).round(2)
speed_summary.columns = ["mean", "median", "std", "pct_inward_or_zero(%)"]
print(speed_summary)

combined = summary.join(speed_summary, how="left")
combined.to_csv(args.out_csv, encoding="utf-8-sig")
print(f"\nsaved table: {args.out_csv}")

# 「弱い外向き/内向き」粒子(=境界際で往復してる疑いがある集団)の規模
weak = df_v[df_v["outward_speed_km_per_day"] <= 5]  # 5km/day以下 = ほぼ静止〜内向き
print(f"\n外向き速度<=5km/日(境界際で往復している疑いがある粒子): "
      f"{len(weak)} / {len(df_v)} ({len(weak)/len(df_v)*100:.1f}% of exited particles, "
      f"{len(weak)/n_particles*100:.2f}% of all particles)")

# ---- 可視化 ----
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

ax = axes[0]
counts = df["edge"].value_counts().reindex(edge_names).fillna(0)
artifact_counts = df[df["likely_artifact"]]["edge"].value_counts().reindex(edge_names).fillna(0)
genuine_counts = counts - artifact_counts
ax.bar(range(len(counts)), genuine_counts.values, color="#1f77b4", label="Genuine (origin far from edge)")
ax.bar(range(len(counts)), artifact_counts.values, bottom=genuine_counts.values,
       color="#d62728", label="Likely artifact (origin <1deg from edge)")
ax.set_xticks(range(len(counts)))
ax.set_xticklabels([e.split("(")[0] for e in counts.index], rotation=20)
ax.set_ylabel("Number of exited particles")
ax.set_title(f"Which edge did particles exit from? (n={len(idx)})")
ax.legend(fontsize=8)

ax = axes[1]
for e, c in zip(edge_names, ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]):
    sub = df_v[df_v["edge"] == e]["outward_speed_km_per_day"]
    if len(sub) > 0:
        ax.hist(sub, bins=40, range=(-50, 150), alpha=0.5, label=e.split("(")[0], color=c)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Outward speed at exit (km/day, negative = inward)")
ax.set_ylabel("Count")
ax.set_title("Outward velocity component by exit edge")
ax.legend(fontsize=8)

ax = axes[2]
ax.hist(df[~df["likely_artifact"]]["exit_day"], bins=30, alpha=0.6, label="Genuine", color="#1f77b4")
ax.hist(df[df["likely_artifact"]]["exit_day"], bins=30, alpha=0.6, label="Likely artifact", color="#d62728")
ax.set_xlabel("Day of exit")
ax.set_ylabel("Count")
ax.set_title("When do particles leave the domain?")
ax.legend(fontsize=8)

ax = axes[3]
ax.scatter(df["origin_dist_to_edge_deg"], df["exit_day"], s=4, alpha=0.2, c="#555555")
ax.axvline(1.0, color="red", ls="--", lw=1, label="1 deg threshold")
ax.set_xlabel("Origin's distance to nearest domain edge (deg)")
ax.set_ylabel("Day of exit")
ax.set_title("Origin proximity to edge vs. exit timing")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"saved figure: {args.out}")
