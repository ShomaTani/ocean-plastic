"""
TRACE — 逆モデル用 train / val / test 分割 (地理クラスタ単位 + n_origins層化)

make_split.py と同じ発想(地理クラスタリング + クラスタ単位分割でリークを防ぐ)を
backward_pairs.npzに適用する。それに加えて、n_origins==1(自明な単一origin問題)と
n_origins>=2(複数originの切り分けが本題)を別々にクラスタリング・分割してから
合体させることで、train/val/testそれぞれに両方のタイプが比例して入るようにする。
(素朴に地理クラスタだけで割ると、testに自明ケースばかり/本題ケースばかり
 偏って集まる回があり、実際に前回はtestの平均n_originsだけ明らかに低かった)
"""

import numpy as np
from sklearn.cluster import KMeans

# =====================================================================
# CONFIG
# =====================================================================
IN_FILE  = "backward_pairs.npz"
OUT_FILE = "backward_pairs.npz"

N_CLUSTERS  = 45  # 30だと粒度が粗すぎてtestに0件になる回があったため増やした
SPLIT_RATIO = (0.8, 0.1, 0.1)
SEED        = 42


def assign_split(lon, lat, n_clusters, seed):
    """地理クラスタリング + クラスタ単位のtrain/val/test割り当てを行い、
    (split配列, cluster_id配列) を返す。make_split.pyと同じロジック。"""
    n = len(lon)
    coords = np.column_stack([lon, lat])
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_id = km.fit_predict(coords)

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(n_clusters)
    cluster_size = np.array([(cluster_id == c).sum() for c in range(n_clusters)])
    train_target = int(round(n * SPLIT_RATIO[0]))
    val_target   = int(round(n * SPLIT_RATIO[1]))

    cluster_to_split = {}
    running_train, running_val = 0, 0
    for c in shuffled:
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
    return split, cluster_id


# =====================================================================
# 1. 既存のペアデータを読み込む
# =====================================================================
print("[1] loading backward_pairs.npz ...")
d = dict(np.load(IN_FILE))
lon, lat, n_origins = d["lon"], d["lat"], d["n_origins"]
n_cells = len(lon)
print(f"    n_cells = {n_cells}")

# =====================================================================
# 2. n_origins==1(自明) と n_origins>=2(本題) を別プールに分けて、
#    プールごとに独立してクラスタリング+分割する
# =====================================================================
easy_mask = n_origins == 1
hard_mask = n_origins >= 2
print(f"[2] easy pool (n_origins=1): {easy_mask.sum()} cells, "
      f"hard pool (n_origins>=2): {hard_mask.sum()} cells")

split = np.empty(n_cells, dtype=object)
cluster_id = np.empty(n_cells, dtype=np.int32)

for name, mask in [("easy", easy_mask), ("hard", hard_mask)]:
    n_sub = mask.sum()
    n_clusters_sub = max(2, round(N_CLUSTERS * n_sub / n_cells))
    print(f"    clustering {name} pool ({n_sub} cells) into {n_clusters_sub} groups ...")
    sub_split, sub_cluster = assign_split(lon[mask], lat[mask], n_clusters_sub, SEED)
    split[mask] = sub_split
    # easy/hardでcluster_idが被らないよう、hardプールのIDはオフセットする
    offset = 0 if name == "easy" else 1000
    cluster_id[mask] = sub_cluster + offset

split = split.astype(str)

# =====================================================================
# 3. 保存 & 内訳を表示
# =====================================================================
d["split"] = split
d["cluster_id"] = cluster_id
np.savez(OUT_FILE, **d)

for s in ("train", "val", "test"):
    m = split == s
    print(f"    {s:5s}: {m.sum():3d} cells  (avg n_origins={n_origins[m].mean():.2f}, "
          f"easy={((m) & easy_mask).sum()}, hard={((m) & hard_mask).sum()})")
print(f"    saved {OUT_FILE} with 'split' and 'cluster_id' added")
