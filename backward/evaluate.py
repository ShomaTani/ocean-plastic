
# 逆モデルのテストセットでの最終評価


import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BackwardDataset
from losses import cross_entropy_loss, to_prob_map, weighted_mse_loss
from main import AttentionUNet
from train import ALPHA, BASE_CH, CKPT_PATH, COASTAL_MASK, DEVICE, IN_CH

test_ds = BackwardDataset("test")
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
print(f"test: {len(test_ds)} samples")

model = AttentionUNet(in_ch=IN_CH, base_ch=BASE_CH).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

total_mse, total_ce, n = 0.0, 0.0, 0
with torch.no_grad():
    for x, y, valid in test_loader:
        x, y, valid = x.to(DEVICE), y.to(DEVICE), valid.to(DEVICE)
        logits = model(x)
        total_mse += weighted_mse_loss(logits, y, alpha=ALPHA, mask=COASTAL_MASK).item() * x.size(0)
        total_ce += cross_entropy_loss(logits, y, valid=valid, mask=COASTAL_MASK).item() * x.size(0)
        n += x.size(0)

print(f"test weighted_mse : {total_mse / n:.6f}")
print(f"test cross_entropy: {total_ce / n:.6f}")

# 目視確認用: test set全地点で 入力(観測点)/予測(責任マップ)/正解 を並べて保存する

import matplotlib.pyplot as plt

x_all, y_all, valid_all = next(iter(DataLoader(test_ds, batch_size=len(test_ds))))
with torch.no_grad():
    logits_all = model(x_all.to(DEVICE))
    prob_all = to_prob_map(logits_all, mask=COASTAL_MASK).cpu().numpy()

n_show = len(test_ds)
fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
for row in range(n_show):
    n_o = test_ds.n_origins[row]
    axes[row, 0].imshow(x_all[row, 0], origin="lower", cmap="viridis")
    axes[row, 0].set_title(f"observation (n_origins={n_o})")
    axes[row, 1].imshow(np.log1p(prob_all[row, 0] * 1000), origin="lower", cmap="viridis")
    axes[row, 1].set_title("predicted responsibility (log)")
    axes[row, 2].imshow(np.log1p(y_all[row, 0].numpy() * 1000), origin="lower", cmap="viridis")
    axes[row, 2].set_title("true responsibility (log)")
plt.tight_layout()
plt.savefig("test_predictions.png", dpi=100)
print(f"saved test_predictions.png ({n_show} cells)")
