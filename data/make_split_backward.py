"""
TRACE — 逆モデル用 train / val / test 分割 (地理クラスタ単位 + n_origins層化)

make_split.py と同じ発想(地理クラスタリング + クラスタ単位分割でリークを防ぐ)を
backward_pairs.npzに適用する。それに加えて、n_origins==1(自明な単一origin問題)と
n_origins>=2(複数originの切り分けが本題)を別々にクラスタリング・分割してから
合体させることで、train/val/testそれぞれに両方のタイプが比例して入るようにする。
(素朴に地理クラスタだけで割ると、testに自明ケースばかり/本題ケースばかり
 偏って集まる回があり、実際に前回はtestの平均n_originsだけ明らかに低かった)

季節データ拡張(build_backward_pairs.py参照)により、同じ観測セル(row, col)が
複数の放出日(年)分のサンプルとして重複しうる。同じセルが年違いでtrain/testに
またがるとリークになるため、(row, col) 単位でユニーク化してからクラスタリング
する。n_origins は年によって変わりうる(同じセルでもある年は1originのみ、
別の年は2origin以上、ということがある)ため、easy/hardプールの判定は
「そのセルが5年間で一度でもn_origins>=2だったか」で決める(=hard優先。
一度でも複数origin問題になり得るセルは、より本題に近い"hard"として扱う)。
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
# 1. 既存のペアデータを読み込み、(row, col) 単位でユニーク化する
# =====================================================================
print("[1] loading backward_pairs.npz ...")
d = dict(np.load(IN_FILE))
row, col, lon, lat, n_origins = d["row"], d["col"], d["lon"], d["lat"], d["n_origins"]
n_rows = len(row)

cell_key = row.astype(np.int64) * 100000 + col.astype(np.int64)  # (row,col)を1本のキーに
uniq_key, first_idx = np.unique(cell_key, return_index=True)
uniq_lon, uniq_lat = lon[first_idx], lat[first_idx]
n_cells = len(uniq_key)
print(f"    n_rows = {n_rows} (ユニークセル数 = {n_cells}, "
      f"複数年で観測されたセル = {n_rows - n_cells})")

# セルごとの「5年間で一度でもn_origins>=2だったか」を集計してhard/easyを決める
uniq_hard = np.zeros(n_cells, dtype=bool)
for i, key in enumerate(uniq_key):
    uniq_hard[i] = (n_origins[cell_key == key] >= 2).any()
print(f"    hard(一度でも複数origin) = {uniq_hard.sum()}, easy(全年で単一origin) = {(~uniq_hard).sum()}")

# =====================================================================
# 2. ユニークセルを easy/hard プールに分け、プールごとに独立してクラスタリング+分割する
# =====================================================================
cell_split = np.empty(n_cells, dtype=object)
cell_cluster = np.empty(n_cells, dtype=np.int32)

for name, mask in [("easy", ~uniq_hard), ("hard", uniq_hard)]:
    n_sub = mask.sum()
    n_clusters_sub = max(2, round(N_CLUSTERS * n_sub / n_cells))
    print(f"[2] clustering {name} pool ({n_sub} unique cells) into {n_clusters_sub} groups ...")
    sub_split, sub_cluster = assign_split(uniq_lon[mask], uniq_lat[mask], n_clusters_sub, SEED)
    cell_split[mask] = sub_split
    offset = 0 if name == "easy" else 1000
    cell_cluster[mask] = sub_cluster + offset

# =====================================================================
# 3. ユニークセル -> split/cluster_id のマッピングを全放出日分の行に配り直す
#    (同じセルは放出日が違っても必ず同じsplitに入る = リーク防止)
# =====================================================================
key_to_split = dict(zip(uniq_key, cell_split))
key_to_cluster = dict(zip(uniq_key, cell_cluster))
split = np.array([key_to_split[k] for k in cell_key]).astype(str)
cluster_id = np.array([key_to_cluster[k] for k in cell_key])

# =====================================================================
# 4. 保存 & 内訳を表示
# =====================================================================
d["split"] = split
d["cluster_id"] = cluster_id
np.savez(OUT_FILE, **d)

hard_row_mask = n_origins >= 2
for s in ("train", "val", "test"):
    m = split == s
    n_cells_in_split = len(set(cell_key[m]))
    print(f"    {s:5s}: {m.sum():4d} rows / {n_cells_in_split} unique cells  "
          f"(avg n_origins={n_origins[m].mean():.2f}, "
          f"hard_rows={(m & hard_row_mask).sum()}, easy_rows={(m & ~hard_row_mask).sum()})")
print(f"    saved {OUT_FILE} with 'split' and 'cluster_id' added")
