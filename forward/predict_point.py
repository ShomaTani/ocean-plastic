"""
TRACE — 順モデル: 任意の1点(放出源)から漂着分布を予測して可視化する

300地点の学習データに含まれない、任意の(lon, lat)を指定して推論できるか
確認するための最小デモ。Streamlitの「地図をクリック→予測」の核になる部分。
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from main import AttentionUNet
from losses import to_prob_map
from train import IN_CH, BASE_CH, CKPT_PATH, COASTAL_MASK, DEVICE

# =====================================================================
# CONFIG — ここを書き換えれば任意の座標で試せる
# =====================================================================
LON, LAT = 129.0234375, 35.05859375  # 釜山沖の海セル(release_sites_used.csvには無い地点)
GRID_N = 128
SIGMA_PX = 1.0
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# =====================================================================
# 1. グリッド定義とcurrent fieldを読み込む
# =====================================================================
print("[1] loading grid definition and current field ...")
ds0 = xr.open_dataset(DATA_DIR / "raw" / "glorys_2018_2022_surface_uovo.nc")
lons = ds0["longitude"].values
lats = ds0["latitude"].values
gx = np.linspace(float(lons.min()), float(lons.max()), GRID_N + 1)
gy = np.linspace(float(lats.min()), float(lats.max()), GRID_N + 1)

c = np.load(DATA_DIR / "current_field.npz")
u, v, land_mask = c["u"], c["v"], c["land_mask"]

# =====================================================================
# 2. (LON, LAT) をピクセルに変換し、陸セルなら警告する
# =====================================================================
col = int(np.clip(np.floor((LON - gx[0]) / (gx[1] - gx[0])), 0, GRID_N - 1))
row = int(np.clip(np.floor((LAT - gy[0]) / (gy[1] - gy[0])), 0, GRID_N - 1))
print(f"    ({LON}, {LAT}) -> pixel (row={row}, col={col}), land={bool(land_mask[row, col])}")
if land_mask[row, col]:
    print("    WARNING: 陸セルです。近傍の海セルを指定し直してください。")

# =====================================================================
# 3. 入力を組み立てる (inputdata_dim.pyと同じ手法)
# =====================================================================
print("[2] building input ...")
onehot = np.zeros((GRID_N, GRID_N), dtype=np.float64)
onehot[row, col] = 1.0
g = gaussian_filter(onehot, sigma=SIGMA_PX, mode="constant")
gaussian = (g / g.sum()).astype(np.float32)

x = torch.from_numpy(np.stack([gaussian, u, v])).unsqueeze(0).float()  # (1,3,128,128)

# =====================================================================
# 4. モデルをロードして推論
# =====================================================================
print("[3] loading model and predicting ...")
model = AttentionUNet(in_ch=IN_CH, base_ch=BASE_CH).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
with torch.no_grad():
    prob = to_prob_map(model(x.to(DEVICE)), mask=COASTAL_MASK).cpu().numpy()[0, 0]  # (128,128)

# =====================================================================
# 5. 可視化
#    「0」と「0に近い低い値」が同じ暗い色になり、分布の裾野が見えなくなる
#    問題への対応: 閾値以下は白、陸地はベージュ、閾値より上だけカラーマップの
#    グラデーションを使う、というRGBA画像を自分で組み立てる。
#
#    predict確率(softmax出力)は理論上どのマスも完全な0にはならないので、
#    「値<=0」を閾値にしても白くならない。「一様分布(何も学習してない状態)なら
#    ここに来るはずの値」(=1/16384)を閾値にすることで、「一様よりは何か
#    予測できているマス」だけを可視化する、という意味のある基準にしている。
#
#    色は「量(確率の大小)」を表すので、dataviz skillの方針通り単色の
#    明→暗グラデーション(sequential)を使う。「濃い紫=深刻」という発想は
#    matplotlibの"Purples"(薄紫→濃紫の単色ランプ)で実現しつつ、陸地は
#    紫と紛れない暖色系(ベージュ)にして、データと地図要素を確実に区別する。
# =====================================================================
LAND_COLOR = (0.87, 0.83, 0.72, 1)  # ベージュ(紫のデータと混同しない暖色系)


CMAP_FLOOR = 0.35  # Purplesの下位35%(白に近すぎる部分)は使わない


def to_rgba(values, land_mask, threshold, cmap_name="Purples", cmap_floor=CMAP_FLOOR):
    below = values <= threshold
    show = ~below & ~land_mask
    cmap = plt.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name
    vmax = values[show].max() if show.any() else 1.0
    norm = plt.Normalize(vmin=threshold, vmax=vmax)
    # 0(白に近すぎる)を避けて、[cmap_floor, 1.0]の範囲だけを使うようにずらす。
    # これで「閾値をわずかに超えただけの値」でも、白とはっきり見分けられる
    # 濃さから始まるようになる。
    t = norm(values)
    rgba = cmap(cmap_floor + (1 - cmap_floor) * t)
    rgba[below] = (1, 1, 1, 1)      # 閾値以下 -> 白
    rgba[land_mask] = LAND_COLOR    # 陸 -> ベージュ
    return rgba, norm, cmap

gaussian_rgba, _, _ = to_rgba(gaussian, land_mask, threshold=0.0)

uniform_baseline = 1.0 / (GRID_N * GRID_N)  # 一様分布ならこの値になるはず
prob_log = np.log1p(prob * 1000)
prob_rgba, prob_norm, prob_cmap = to_rgba(
    prob_log, land_mask, threshold=np.log1p(uniform_baseline * 1000)
)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(gaussian_rgba, origin="lower")
axes[0].set_title(f"release point\nlon={LON:.2f} lat={LAT:.2f}")

axes[1].imshow(prob_rgba, origin="lower")
axes[1].set_title("predicted beaching distribution (log scale)")
sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
plt.colorbar(sm, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.savefig("predict_point_output.png", dpi=110)
print("saved predict_point_output.png")
