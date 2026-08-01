"""
越境輸送(日本海横断など)が起きているかを数値で検証する。

集約マップでは個々の「どこからどこへ」が埋もれてしまうため、
origin(放出源)の所属地域と、漂着先の所属地域を突き合わせて
輸送行列(origin地域 → 漂着地域)を作る。

環境省の実測(対馬等での韓国語ポリタンク比率)と比較するための
土台にもなる。

使い方:
  python check_cross_transport.py trace_run.zarr
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr

zarr_path = sys.argv[1] if len(sys.argv) > 1 else "trace_run.zarr"

ds = xr.open_zarr(zarr_path)
flon = ds.lon.isel(obs=-1).values
flat = ds.lat.isel(obs=-1).values
slon = ds.lon.isel(obs=0).values
slat = ds.lat.isel(obs=0).values
beached = ds.beached.isel(obs=-1).values
left_domain = ds.left_domain.isel(obs=-1).values
origin = ds.origin.values


def region_of(lon, lat):
    """粗い地域分類。厳密な国境ではなく、輸送方向を掴むための区分。"""
    # 日本(本州・四国・九州・北海道をまとめる): おおむね東経128度以東かつ
    # 日本列島の緯度帯。ただし朝鮮半島(東経126-130, 北緯34-43)を除く。
    if 126.0 <= lon <= 130.0 and 34.0 <= lat <= 43.5:
        return "Korea"
    if lon < 126.0:
        if lat >= 40.0:
            return "China_NE/Russia"
        return "China"
    if lon >= 129.0:
        return "Japan"
    return "Other"


start_region = np.array([region_of(a, b) for a, b in zip(slon, slat)])
end_region = np.array([region_of(a, b) for a, b in zip(flon, flat)])

# 実際に漂着した粒子のみ(境界離脱は輸送先が不明なので除外)
valid = (beached == 1) & (left_domain == 0)
print(f"総粒子数: {len(origin)}")
print(f"うち漂着(境界離脱を除く): {valid.sum()}\n")

df = pd.DataFrame({
    "from": start_region[valid],
    "to": end_region[valid],
})
matrix = pd.crosstab(df["from"], df["to"])
print("=== 輸送行列 (行=放出源地域, 列=漂着地域, 単位=粒子数) ===")
print(matrix)
print()

# 行ごとの割合
print("=== 同じものを行方向の割合(%)で ===")
print((matrix.T / matrix.sum(axis=1) * 100).T.round(1))
print()

# 越境した粒子だけ抜き出す
cross = df[df["from"] != df["to"]]
print(f"越境輸送した粒子: {len(cross)} / {valid.sum()} "
      f"({len(cross)/valid.sum()*100:.1f}%)")
print()

# 特に注目: 韓国・中国 → 日本
k2j = ((df["from"] == "Korea") & (df["to"] == "Japan")).sum()
c2j = ((df["from"] == "China") & (df["to"] == "Japan")).sum()
j2j = ((df["from"] == "Japan") & (df["to"] == "Japan")).sum()
to_japan = (df["to"] == "Japan").sum()
print("=== 日本に漂着した粒子の起源内訳 ===")
print(f"  日本沿岸から      : {j2j:6d} ({j2j/to_japan*100:5.1f}%)" if to_japan else "  (日本漂着なし)")
if to_japan:
    print(f"  韓国沿岸から      : {k2j:6d} ({k2j/to_japan*100:5.1f}%)")
    print(f"  中国沿岸から      : {c2j:6d} ({c2j/to_japan*100:5.1f}%)")
    print()
    print("  ※環境省の廃ポリタンク調査では、言語表記から判読できたものの")
    print("    大半が韓国語表記だった。上の韓国由来比率が極端に低ければ、")
    print("    windage係数や放流点配置の再検討が必要。")

# 移動距離の分布
disp = np.hypot(flon - slon, flat - slat)[valid]
print()
print("=== 漂着粒子の移動距離分布 (度) ===")
for q in [50, 75, 90, 95, 99]:
    print(f"  {q}パーセンタイル: {np.percentile(disp, q):.2f} 度 "
          f"(約{np.percentile(disp, q)*111:.0f} km)")
print(f"  最大            : {disp.max():.2f} 度 (約{disp.max()*111:.0f} km)")
print()
print("  ※日本海横断には概ね2〜6度(200〜700km)の移動が必要。")
print("    90パーセンタイルがこれを大きく下回るなら横断は稀ということ。")