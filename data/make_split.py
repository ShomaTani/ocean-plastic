"""
TRACE — train / val / test 分割 (地理クラスタ単位の Grouped Split)

train_pairs.npz (inputdata_dim.py の出力) に "split" 列を追加して保存し直す。
個々の地点をバラバラにランダム分割すると、近接origin同士(似た海流パターン)が
train/valにまたがってリークするおそれがあるため、
  1) 地点を地理的にクラスタリング
  2) クラスタを丸ごと train/val/test のどれかに割り当てる
という2段階でリークを避けつつ、クラスタ数を多めに取ることで地域的な偏りも抑える。

季節データ拡張(inputdata_dim.py参照)により、同じsite_idが複数の放出日(年)分
複製されている。同じ地点が年違いでtrain/testにまたがると「同じ場所を答えの
一部を見た状態でテストする」リークになるため、クラスタリングは
site_id単位でユニーク化してから行い、結果を全放出日の行に配り直す。

パラメータ(N_CLUSTERS, SPLIT_RATIO, SEED)は暫定値。チューニングは別途行う想定。
"""

import numpy as np
from sklearn.cluster import KMeans

# =====================================================================
# CONFIG — 暫定値。後で調整する
# =====================================================================
IN_FILE  = "train_pairs.npz"
OUT_FILE = "train_pairs.npz"   # 上書き保存(splitを追加した完全版に置き換える)

N_CLUSTERS  = 32            # 500地点に増量した際、18だと粗すぎてtestが0件になったため増やした
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test の目標割合(クラスタ単位なので厳密には一致しない)
SEED        = 42            # KMeansの初期化とクラスタのシャッフルを固定するための乱数シード

# =====================================================================
# 1. 既存のペアデータを読み込み、site_id単位でユニーク化する
# =====================================================================
print("[1] loading train_pairs.npz ...")
d = dict(np.load(IN_FILE))
site_id, lon, lat = d["site_id"], d["lon"], d["lat"]
n_rows = len(site_id)

uniq_site_id, first_idx = np.unique(site_id, return_index=True)
uniq_lon, uniq_lat = lon[first_idx], lat[first_idx]
n_sites = len(uniq_site_id)
print(f"    n_rows = {n_rows} (site_id unique数 = {n_sites}, "
      f"1siteあたり{n_rows // n_sites}放出日分)")

# =====================================================================
# 2. ユニーク地点を地理的にクラスタリングする
#    KMeansは (lon, lat) をそのままユークリッド距離で見るので、本来は
#    緯度による経度方向のスケール歪み(cos(lat)分だけ経度1度の実距離が変わる)
#    を補正すべきだが、大まかな地域分けが目的なのでここでは無視する
# =====================================================================
print(f"[2] clustering {n_sites} unique sites into {N_CLUSTERS} groups ...")
coords = np.column_stack([uniq_lon, uniq_lat])
km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
uniq_cluster_id = km.fit_predict(coords)   # 各ユニーク地点のクラスタ (n_sites,)

# =====================================================================
# 3. クラスタを丸ごと train / val / test に割り振る(site_id単位)
# =====================================================================
print("[3] assigning clusters to train/val/test ...")
rng = np.random.RandomState(SEED)
cluster_ids_shuffled = rng.permutation(N_CLUSTERS)

cluster_size = np.array([(uniq_cluster_id == c).sum() for c in range(N_CLUSTERS)])
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

site_to_split = {sid: cluster_to_split[c] for sid, c in zip(uniq_site_id, uniq_cluster_id)}
site_to_cluster = {sid: c for sid, c in zip(uniq_site_id, uniq_cluster_id)}

# =====================================================================
# 4. site_id -> split/cluster_id のマッピングを全放出日分の行に配り直す
#    (同じsite_idは放出日が違っても必ず同じsplitに入る = リーク防止)
# =====================================================================
split = np.array([site_to_split[sid] for sid in site_id])
cluster_id = np.array([site_to_cluster[sid] for sid in site_id])

# =====================================================================
# 5. 保存 & 内訳を表示
# =====================================================================
d["split"] = split
d["cluster_id"] = cluster_id
np.savez(OUT_FILE, **d)

valid = d["valid"]
for s in ("train", "val", "test"):
    m = split == s
    n_sites_in_split = len(set(site_id[m]))
    print(f"    {s:5s}: {m.sum():3d} rows / {n_sites_in_split} unique sites  "
          f"(valid={valid[m].sum()}, zero-beach={(~valid[m]).sum()})")
print(f"    saved {OUT_FILE} with 'split' and 'cluster_id' added")
