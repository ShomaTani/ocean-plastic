"""
密度マップの中身を数値で確認する診断スクリプト。

使い方: trace_sim_test.py と同じディレクトリ(=npyファイルがある場所)で実行
  python inspect_density.py [--origin 0]
"""
import argparse
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--origin", type=int, default=None,
               help="調べるorigin番号。省略時は自動選択されたものと同じロジックで選ぶ")
args = p.parse_args()

maps_all   = np.load("density_maps_all.npy")    # (n_sites, GRID_N, GRID_N)
maps_beach = np.load("density_maps_beach.npy")
sites = pd.read_csv("release_sites_used.csv")

n_sites, GRID_N, _ = maps_all.shape
lons = np.linspace(sites["lon"].min(), sites["lon"].max(), GRID_N)  # 近似、正確なgx/gyが必要なら要調整
print(f"n_sites={n_sites}, grid={GRID_N}x{GRID_N}")

k = args.origin if args.origin is not None else 0
site_lon, site_lat = sites.loc[k, "lon"], sites.loc[k, "lat"]
print(f"\n--- origin {k}  (release point: lon={site_lon:.2f}, lat={site_lat:.2f}) ---")

for name, H in [("all", maps_all[k]), ("beach", maps_beach[k])]:
    nz = np.argwhere(H > 0)
    print(f"\n[{name}]")
    print(f"  sum          : {H.sum():.4f}")
    print(f"  nonzero cells: {len(nz)} / {GRID_N*GRID_N}")
    print(f"  max value    : {H.max():.4f}  (1セルにこの割合が集中)")
    if len(nz) > 0:
        # 上位5セルの値と、放流地点からの近似距離(グリッドindex差)
        flat_idx = np.argsort(H.ravel())[::-1][:5]
        rows, cols = np.unravel_index(flat_idx, H.shape)
        print(f"  上位{min(5,len(nz))}セル (row=lat方向, col=lon方向 のグリッドindex, 値):")
        for r, c in zip(rows, cols):
            if H[r, c] > 0:
                print(f"    (row={r:3d}, col={c:3d})  value={H[r,c]:.4f}")
        # グリッド全体の何%の位置にあるか(0=左下端, 1=右上端)
        print(f"  → row/col が 0 や {GRID_N-1} に近いほど、地図の端(=放流点付近)に張り付いている")

print("\n診断:")
Hshow = maps_all[k]
if Hshow.max() > 0.3 and (Hshow > 0).sum() <= 3:
    print("  → ごく少数のセルに集中。おそらく放流点のすぐ隣で即座に漂着している。")
    print("     このoriginでは粒子がほとんど拡散していない可能性が高い。")
    print("     RUNTIME_DAYSが短すぎるか、この地点が入り江・湾内で移流が弱い場所かも。")
elif Hshow.sum() == 0:
    print("  → 完全にゼロ。粒子がこのoriginに割り当てられていない(land-skip等)。")
else:
    print("  → 複数セルに分散している。プロットのvmax/解像度の問題で見えにくいだけかも。")