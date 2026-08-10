"""
TRACE — 順モデル(Attention U-Net)の訓練ループ

CONFIGの値(BATCH_SIZE, LR, EPOCHS, ALPHA等)は暫定値。チューニングは別途行う。
LOSS_FN で cross_entropy / weighted_mse を切り替えられるようにしてある。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TraceDataset
from main import AttentionUNet
from losses import cross_entropy_loss, weighted_mse_loss

# =====================================================================
# CONFIG — 暫定値
# =====================================================================
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 100
BASE_CH = 32
LOSS_FN = "weighted_mse"   # "weighted_mse" or "cross_entropy"
ALPHA = 5.0                # weighted_mse用の非ゼロ画素の重み
PATIENCE = 15              # val lossがこの回数連続で改善しなければ早期終了
CKPT_PATH = Path(__file__).resolve().parent / "best_model.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def compute_loss(logits, y, valid):
    if LOSS_FN == "weighted_mse":
        return weighted_mse_loss(logits, y, alpha=ALPHA)
    elif LOSS_FN == "cross_entropy":
        return cross_entropy_loss(logits, y, valid=valid)
    raise ValueError(f"unknown LOSS_FN: {LOSS_FN}")


def run_epoch(model, loader, optimizer=None):
    """optimizerがNoneならeval(勾配なし)、渡されればtrain(逆伝播あり)。
    train/evalを1つの関数にまとめて、訓練ループ側の重複を減らしている。"""
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

            total_loss += loss.item() * x.size(0)  # バッチ平均 x サンプル数 = 合計
            n += x.size(0)
    return total_loss / n  # 全サンプル平均に戻す


def main():
    print(f"device: {DEVICE}")

    train_ds = TraceDataset("train")
    val_ds = TraceDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    model = AttentionUNet(base_ch=BASE_CH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    epochs_since_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, optimizer=None)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            epochs_since_improve += 1

        print(f"epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}"
              f"{'  * best' if improved else ''}")

        if epochs_since_improve >= PATIENCE:
            print(f"early stopping (val loss未改善が{PATIENCE}epoch続いた)")
            break

    print(f"done. best val loss = {best_val:.6f}  (saved to {CKPT_PATH})")


if __name__ == "__main__":
    main()
