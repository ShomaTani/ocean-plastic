"""
TRACE — train / val / test 分割 (地理クラスタ単位の Grouped Split)

train_pairs.npz (inputdata_dim.py の出力) に "split" 列を追加して保存し直す。
個々の地点をバラバラにランダム分割すると、近接origin同士(似た海流パターン)が
train/valにまたがってリークするおそれがあるため、
  1) 地点を地理的にクラスタリング
  2) クラスタを丸ごと train/val/test のどれかに割り当てる
という2段階でリークを避けつつ、クラスタ数を多めに取ることで地域的な偏りも抑える。

パラメータ(N_CLUSTERS, SPLIT_RATIO, SEED)は暫定値。チューニングは別途行う想定。

Splits the training, test and validation data
"""

import numpy as np
from sklearn.cluster import KMeans

# =====================================================================
# CONFIG — 暫定値。後で調整する
# =====================================================================
IN_FILE  = "train_pairs.npz"
OUT_FILE = "train_pairs.npz"   # 上書き保存(splitを追加した完全版に置き換える)

N_CLUSTERS  = 18            # 地理クラスタ数。多いほど地域偏りは減るがクラスタが小さくなる
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test の目標割合(クラスタ単位なので厳密には一致しない)
SEED        = 42            # KMeansの初期化とクラスタのシャッフルを固定するための乱数シード

# =====================================================================
# 1. 既存のペアデータを読み込む
# =====================================================================
print("[1] loading train_pairs.npz ...")
d = dict(np.load(IN_FILE))
lon, lat = d["lon"], d["lat"]
n_sites = len(lon)
print(f"    n_sites = {n_sites}")

# =====================================================================
# 2. 地点を地理的にクラスタリングする
#    KMeansは (lon, lat) をそのままユークリッド距離で見るので、本来は
#    緯度による経度方向のスケール歪み(cos(lat)分だけ経度1度の実距離が変わる)
#    を補正すべきだが、大まかな地域分けが目的なのでここでは無視する
# =====================================================================
print(f"[2] clustering {n_sites} sites into {N_CLUSTERS} groups ...")
coords = np.column_stack([lon, lat])
km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
cluster_id = km.fit_predict(coords)   # 各地点がどのクラスタに属すか (n_sites,)

# =====================================================================
# 3. クラスタを丸ごと train / val / test に割り振る
#    - クラスタの並び順をシャッフルしてから、目標サンプル数に達するまで
#      順番に train → val → test へクラスタを積んでいく
#    - クラスタ単位なので比率はSPLIT_RATIOにきっちり一致はしない(小規模データの限界)
# =====================================================================
print("[3] assigning clusters to train/val/test ...")
rng = np.random.RandomState(SEED)
cluster_ids_shuffled = rng.permutation(N_CLUSTERS)

cluster_size = np.array([(cluster_id == c).sum() for c in range(N_CLUSTERS)])
train_target = int(round(n_sites * SPLIT_RATIO[0]))
val_target   = int(round(n_sites * SPLIT_RATIO[1]))
# test_target は残り全部(丸め誤差をここに吸収させる)

cluster_to_split = {}
running_train, running_val = 0, 0
for c in cluster_ids_shuffled:
    size = cluster_size[c]
    if running_train < train_target:
        cluster_to_split[c] = "train"
        running_train += size
    elif running_val < val_target:
        cluster_to_split[c] = "val"
        running_val += size
    else:
        cluster_to_split[c] = "test"

split = np.array([cluster_to_split[c] for c in cluster_id])

# =====================================================================
# 4. 保存 & 内訳を表示
# =====================================================================
d["split"] = split
d["cluster_id"] = cluster_id
np.savez(OUT_FILE, **d)

valid = d["valid"]
for s in ("train", "val", "test"):
    m = split == s
    print(f"    {s:5s}: {m.sum():3d} sites  (valid={valid[m].sum()}, zero-beach={(~valid[m]).sum()})")
print(f"    saved {OUT_FILE} with 'split' and 'cluster_id' added")

