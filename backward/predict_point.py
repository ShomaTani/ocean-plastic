"""
TRACE — 逆モデル: 任意の1点(観測地点)から責任マップを予測して可視化する

617観測セルの学習データに含まれない、任意の(lon, lat)で「ここでプラスチックが
見つかったら、どこのoriginが疑わしいか」を推論するデモ。forward/predict_point.py
と対になるスクリプトで、可視化まわり(陸地のベージュ塗り, 閾値, 単色紫グラデーション)
は共通のロジックをそのまま踏襲している。
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from dataset import load_coastal_mask
from main import AttentionUNet
from losses import to_prob_map
from train import IN_CH, BASE_CH, CKPT_PATH, DEVICE

# =====================================================================
# CONFIG — ここを書き換えれば任意の座標で試せる
# =====================================================================
LON, LAT = 130.4, 33.6  # 福岡沖の海セル(backward_pairs.npzの617セルには無い地点)
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
# 2. (LON, LAT) をピクセルに変換
# =====================================================================
col = int(np.clip(np.floor((LON - gx[0]) / (gx[1] - gx[0])), 0, GRID_N - 1))
row = int(np.clip(np.floor((LAT - gy[0]) / (gy[1] - gy[0])), 0, GRID_N - 1))
print(f"    ({LON}, {LAT}) -> pixel (row={row}, col={col}), land={bool(land_mask[row, col])}")
if land_mask[row, col]:
    print("    WARNING: 陸セルです。近傍の海セルを指定し直してください。")

# =====================================================================
# 3. 入力(観測点ガウシアン + 海流)を組み立てる
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

# bexp07以降、originは必ず沿岸(陸に接した海セル)という制約がmasked softmaxとして
# モデル自体に組み込まれている(訓練時と同じマスクを推論時にも渡す)。
coastal_mask = load_coastal_mask().to(DEVICE)
with torch.no_grad():
    prob = to_prob_map(model(x.to(DEVICE)), mask=coastal_mask).cpu().numpy()[0, 0]  # (128,128)

# =====================================================================
# 5. 可視化(forward/predict_point.pyと同じ配色ロジック)
#    陸地はベージュ、閾値(一様分布の基準値)以下は白、それより上だけ
#    Purples(単色, 明→暗)のグラデーションを使う。下位35%は白に近すぎて
#    見分けづらいので使わない。
# =====================================================================
LAND_COLOR = (0.87, 0.83, 0.72, 1)
CMAP_FLOOR = 0.35


def to_rgba(values, land_mask, threshold, cmap_name="Purples", cmap_floor=CMAP_FLOOR):
    below = values <= threshold
    show = ~below & ~land_mask
    cmap = plt.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name
    vmax = values[show].max() if show.any() else 1.0
    norm = plt.Normalize(vmin=threshold, vmax=vmax)
    t = norm(values)
    rgba = cmap(cmap_floor + (1 - cmap_floor) * t)
    rgba[below] = (1, 1, 1, 1)
    rgba[land_mask] = LAND_COLOR
    return rgba, norm, cmap

gaussian_rgba, _, _ = to_rgba(gaussian, land_mask, threshold=0.0)

uniform_baseline = 1.0 / (GRID_N * GRID_N)
prob_log = np.log1p(prob * 1000)
prob_rgba, prob_norm, prob_cmap = to_rgba(
    prob_log, land_mask, threshold=np.log1p(uniform_baseline * 1000)
)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(gaussian_rgba, origin="lower")
axes[0].set_title(f"observation point\nlon={LON:.2f} lat={LAT:.2f}")

axes[1].imshow(prob_rgba, origin="lower")
axes[1].set_title("predicted responsibility (log scale)")
sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
plt.colorbar(sm, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.savefig("predict_point_output.png", dpi=110)
print("saved predict_point_output.png")
