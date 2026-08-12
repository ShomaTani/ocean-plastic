"""
TRACE — 逆モデル用 train / val / test 分割 (地理クラスタ単位の Grouped Split)

make_split.py と同じ発想(地点を地理クラスタリングしてクラスタ単位で分割、
近接地点が train/val にまたがるリークを防ぐ)を backward_pairs.npz に適用する。
"""

import numpy as np
from sklearn.cluster import KMeans

# =====================================================================
# CONFIG
# =====================================================================
IN_FILE  = "backward_pairs.npz"
OUT_FILE = "backward_pairs.npz"

N_CLUSTERS  = 30            # 617地点 / 300地点 の比率に合わせてmake_split.pyの18から増やした
SPLIT_RATIO = (0.8, 0.1, 0.1)
SEED        = 42

# =====================================================================
# 1. 既存のペアデータを読み込む
# =====================================================================
print("[1] loading backward_pairs.npz ...")
d = dict(np.load(IN_FILE))
lon, lat = d["lon"], d["lat"]
n_cells = len(lon)
print(f"    n_cells = {n_cells}")

# =====================================================================
# 2. 観測セルを地理的にクラスタリングする
# =====================================================================
print(f"[2] clustering {n_cells} cells into {N_CLUSTERS} groups ...")
coords = np.column_stack([lon, lat])
km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
cluster_id = km.fit_predict(coords)

# =====================================================================
# 3. クラスタを丸ごと train / val / test に割り振る (make_split.pyと同じロジック)
# =====================================================================
print("[3] assigning clusters to train/val/test ...")
rng = np.random.RandomState(SEED)
cluster_ids_shuffled = rng.permutation(N_CLUSTERS)

cluster_size = np.array([(cluster_id == c).sum() for c in range(N_CLUSTERS)])
train_target = int(round(n_cells * SPLIT_RATIO[0]))
val_target   = int(round(n_cells * SPLIT_RATIO[1]))

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

for s in ("train", "val", "test"):
    m = split == s
    print(f"    {s:5s}: {m.sum():3d} cells  (avg n_origins={d['n_origins'][m].mean():.1f})")
print(f"    saved {OUT_FILE} with 'split' and 'cluster_id' added")
