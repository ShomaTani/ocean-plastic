"""
逆モデルのシミュレータに対する精度を、レポートで説明しやすい物理量(km, %)で定量化。

  centroid_distance_km : 予測した責任マップPと正解の責任マップYの
                          確率重み付き重心同士の距離(km)
  total_variation       : 0.5 * sum(|P - Y|)。0=完全一致, 1=完全不一致
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BackwardDataset
from losses import to_prob_map
from main import AttentionUNet
from train import BASE_CH, CKPT_PATH, COASTAL_MASK, DEVICE, IN_CH

GRID_N = 128
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 120.0, 150.0, 25.0, 50.0
gx = np.linspace(LON_MIN, LON_MAX, GRID_N + 1)
gy = np.linspace(LAT_MIN, LAT_MAX, GRID_N + 1)
cell_lon = (gx[:-1] + gx[1:]) / 2
cell_lat = (gy[:-1] + gy[1:]) / 2
LON_GRID, LAT_GRID = np.meshgrid(cell_lon, cell_lat)
DEG_KM_LAT = 111.0


def weighted_centroid(P):
    s = P.sum()
    if s <= 0:
        return None
    lon_c = (P * LON_GRID).sum() / s
    lat_c = (P * LAT_GRID).sum() / s
    return lon_c, lat_c


def centroid_distance_km(P, Y):
    cp, cy = weighted_centroid(P), weighted_centroid(Y)
    if cp is None or cy is None:
        return None
    dlon = (cp[0] - cy[0]) * DEG_KM_LAT * np.cos(np.radians((cp[1] + cy[1]) / 2))
    dlat = (cp[1] - cy[1]) * DEG_KM_LAT
    return float(np.hypot(dlon, dlat))


def total_variation(P, Y):
    return float(0.5 * np.abs(P - Y).sum())


test_ds = BackwardDataset("test")
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
print(f"test: {len(test_ds)} samples")

model = AttentionUNet(in_ch=IN_CH, base_ch=BASE_CH).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

centroid_dists, tv_dists, n_origins_list = [], [], []
with torch.no_grad():
    for x, y, valid in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        prob = to_prob_map(logits, mask=COASTAL_MASK).cpu().numpy()[:, 0]
        y_np = y.numpy()[:, 0]
        for i in range(len(x)):
            tv_dists.append(total_variation(prob[i], y_np[i]))
            d = centroid_distance_km(prob[i], y_np[i])
            if d is not None:
                centroid_dists.append(d)

centroid_dists = np.array(centroid_dists)
tv_dists = np.array(tv_dists)
n_origins = test_ds.n_origins

print("\n===== シミュレータに対する精度 (test split) =====")
print(f"centroid distance (km): mean={centroid_dists.mean():.1f}  median={np.median(centroid_dists):.1f}  "
      f"p90={np.percentile(centroid_dists, 90):.1f}  n={len(centroid_dists)}")
print(f"total variation (0-1) : mean={tv_dists.mean():.3f}  median={np.median(tv_dists):.3f}  "
      f"p90={np.percentile(tv_dists, 90):.3f}  n={len(tv_dists)}")

# n_origins==1(自明)とn_origins>=2(本題)で分けても見る
easy = n_origins == 1
hard = n_origins >= 2
print(f"\n内訳(n_origins=1, easy, n={easy.sum()}): "
      f"centroid={np.array(centroid_dists)[easy].mean():.1f}km, tv={tv_dists[easy].mean():.3f}")
print(f"内訳(n_origins>=2, hard, n={hard.sum()}): "
      f"centroid={np.array(centroid_dists)[hard].mean():.1f}km, tv={tv_dists[hard].mean():.3f}")
print("====================================================")
