"""
TRACE — 逆モデル(Attention U-Net)の訓練ループ

forward/train.pyと同じ構造。CONFIGの初期値もforwardで実験して見つけた
ベスト設定(LR=3e-3, weight_decay=1e-3, cross_entropy, dropout無し)を
そのまま出発点にしている。ただしタスクが違う(観測点→責任マップ)ので、
この値が逆モデルにも最適とは限らない。再チューニングは別途行う。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BackwardDataset, load_coastal_mask
from main import AttentionUNet
from losses import cross_entropy_loss, weighted_mse_loss

# =====================================================================
# CONFIG — forwardのベスト設定を初期値として流用(要再チューニング)
# =====================================================================
SEED = 42
BATCH_SIZE = 16
LR = 3e-3
WEIGHT_DECAY = 1e-5        # bexp04: sweepで最良(forwardの1e-3はむしろ悪化した)
EPOCHS = 100
IN_CH = 3                  # [観測点ガウシアン, 平均u, 平均v]
BASE_CH = 32
DROPOUT_P = 0.0
LOSS_FN = "cross_entropy"  # "weighted_mse" or "cross_entropy"
ALPHA = 5.0                # weighted_mse用の非ゼロ画素の重み
PATIENCE = 15
USE_SCHEDULER = False
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 5
CKPT_PATH = Path(__file__).resolve().parent / "best_model.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
COASTAL_MASK = load_coastal_mask().to(DEVICE)  # originは必ず沿岸のはず、というmasked softmax制約


def compute_loss(logits, y, valid):
    if LOSS_FN == "weighted_mse":
        return weighted_mse_loss(logits, y, alpha=ALPHA, mask=COASTAL_MASK)
    elif LOSS_FN == "cross_entropy":
        return cross_entropy_loss(logits, y, valid=valid, mask=COASTAL_MASK)
    raise ValueError(f"unknown LOSS_FN: {LOSS_FN}")


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for x, y, valid in loader:
            x, y, valid = x.to(DEVICE), y.to(DEVICE), valid.to(DEVICE)
            logits = model(x)
            loss = compute_loss(logits, y, valid)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / n


def main():
    torch.manual_seed(SEED)
    print(f"device: {DEVICE}, seed: {SEED}")

    train_ds = BackwardDataset("train")
    val_ds = BackwardDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    model = AttentionUNet(in_ch=IN_CH, base_ch=BASE_CH, dropout_p=DROPOUT_P).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
        )

    best_val = float("inf")
    train_at_best = None
    epochs_since_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, optimizer=None)

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            train_at_best = train_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            epochs_since_improve += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.1e}"
              f"{'  * best' if improved else ''}")

        if epochs_since_improve >= PATIENCE:
            print(f"early stopping (val loss未改善が{PATIENCE}epoch続いた)")
            break

    gap = train_at_best is not None and (best_val - train_at_best)
    print(f"done. best val loss = {best_val:.6f}  (train@best={train_at_best:.6f}, "
          f"gap={gap:.6f})  saved to {CKPT_PATH}")
    return best_val, train_at_best


if __name__ == "__main__":
    main()
