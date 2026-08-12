"""
TRACE — 順モデル(Attention U-Net)の訓練ループ

CONFIGの値(BATCH_SIZE, LR, EPOCHS, ALPHA等)は暫定値。チューニングは別途行う。
LOSS_FN で cross_entropy / weighted_mse を切り替えられるようにしてある。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TraceDataset, load_coastal_mask
from main import AttentionUNet
from losses import cross_entropy_loss, weighted_mse_loss

# =====================================================================
# CONFIG — 暫定値
# =====================================================================
SEED = 42                  # 実験間で初期値・shuffleを揃えて比較できるようにする
BATCH_SIZE = 16
LR = 3e-3                  # exp05: LR sweep(3e-4,1e-3,3e-3,1e-2)で最良だった値
WEIGHT_DECAY = 1e-3        # exp08: sweepで最良(val 3.058->2.965, gap 1.78->0.50)
EPOCHS = 100
IN_CH = 3                  # exp07でspeedチャンネルも試したが不採用、3chに戻す
BASE_CH = 32
DROPOUT_P = 0.0            # exp09: 0.1/0.2/0.3を試したが全て悪化(weight_decayと重複して過剰正則化)。0.0を維持
LOSS_FN = "cross_entropy"  # exp04: weighted_mse -> cross_entropy
ALPHA = 5.0                # weighted_mse用の非ゼロ画素の重み(cross_entropy使用中は無視)
PATIENCE = 15               # exp01/exp02と同じ値に戻す
USE_SCHEDULER = False       # exp06で試したが効果なし(val 3.058→3.170と悪化)だったのでOFFに戻す
SCHED_FACTOR = 0.5          # val loss停滞時にLRを何倍にするか
SCHED_PATIENCE = 5          # 何epoch停滞したらLRを下げるか(早期終了のPATIENCEより短くする)
COASTAL_DILATION = 3        # exp10: Streamlitデモで発覚した「海上への謎の広がり」対策。
                             # masked softmaxで非沿岸セルへの確率漏れを抑える(緩め設定)
CKPT_PATH = Path(__file__).resolve().parent / "best_model.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
COASTAL_MASK = load_coastal_mask(dilation=COASTAL_DILATION).to(DEVICE)


def compute_loss(logits, y, valid):
    if LOSS_FN == "weighted_mse":
        return weighted_mse_loss(logits, y, alpha=ALPHA, mask=COASTAL_MASK)
    elif LOSS_FN == "cross_entropy":
        return cross_entropy_loss(logits, y, valid=valid, mask=COASTAL_MASK)
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
    torch.manual_seed(SEED)
    print(f"device: {DEVICE}, seed: {SEED}")

    train_ds = TraceDataset("train")
    val_ds = TraceDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    model = AttentionUNet(in_ch=IN_CH, base_ch=BASE_CH, dropout_p=DROPOUT_P).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if USE_SCHEDULER:
        # val lossがSCHED_PATIENCE epoch連続で改善しなければLRをSCHED_FACTOR倍に下げる
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE
        )

    best_val = float("inf")
    train_at_best = None  # best_valを更新した瞬間のtrain loss(過学習の"gap"を見るため)
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
