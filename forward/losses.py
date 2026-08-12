"""
TRACE — 順モデル用 Loss

AttentionUNet(main.py)はロジット(生の値、活性化なし)を返す設計にしてある。
ここでまず to_prob_map() で「画像全体で合計1になる確率分布」に変換してから、
2種類のlossのどちらかで正解(Y, 合計1の確率分布)と比較する。

  - cross_entropy_loss : 正解が集中してる場所の予測確率を直接最大化する
  - weighted_mse_loss   : 非ゼロ画素の重みを上げた重み付きMSE

どちらもYの極端なスパースさ(ほぼ全マス0)に対する対策。素のMSEだと
「全部0を出す」だけでlossが下がって見える問題があるため(詳細は会話ログ参照)。
"""

import torch


def to_prob_map(logits):
    """モデルの生の出力 (B,1,H,W) を、画像全体で合計1になる確率分布に変換する。
    画素ごとに独立なsigmoidと違い、softmaxは全画素が"取り合い"になるので
    合計が1という制約を厳密に満たせる(=Yと同じ「確率分布」として比較できる)。"""
    B, C, H, W = logits.shape
    flat = logits.view(B, -1)          # (B, H*W)
    prob = torch.softmax(flat, dim=1)  # 合計1
    return prob.view(B, C, H, W)


def cross_entropy_loss(logits, target, valid=None, eps=1e-8):
    """target(合計1の確率分布)に対するクロスエントロピー。

    target=0のピクセルは `target * log(prob)` の項がそのまま0になるので、
    「正解が集中してる数マスに、予測側もどれだけ確率を寄せられてるか」だけが
    lossに効いてくる。

    valid=False のサンプル(90日間漂着なし、targetが全マス0)は、target自体が
    全部0なのでこのlossは自動的に0を返す = 勾配が立たず、学習に寄与しない。
    数値的には壊れないが「そのサンプルからは何も学ばない」のと同じなので、
    validでフィルタして平均を取ることで、レポートするloss値が
    "学習に寄与したサンプルだけの平均" になるようにしている。
    """
    prob = to_prob_map(logits)
    per_sample = -(target * torch.log(prob + eps)).flatten(1).sum(dim=1)  # (B,)
    if valid is not None:
        per_sample = per_sample[valid]
        if per_sample.numel() == 0:
            return per_sample.sum()  # このバッチにvalidなサンプルが1つもない
    return per_sample.mean()


def weighted_mse_loss(logits, target, alpha=5.0, valid=None):
    """target非ゼロ画素の重みをalpha倍にした重み付きMSE。
    予測側もsoftmaxで合計1に揃えてから比較する(スケールをYに合わせるため)。

    cross_entropy_lossと違って、target=0のサンプル(90日間漂着なし)でも
    意味のある学習信号になる: predはsoftmaxでsum=1に固定されてて
    全部0にはできないので、target=0との誤差を最小化しようとすると
    「特定の場所に集中させず薄く広げる」方向に寄っていく
    (=「この起源は特定の場所に偏らない」という信号として機能する)。
    なので通常はvalidを渡さず、invalidサンプルも学習に含めてよい。
    """
    prob = to_prob_map(logits)
    weight = 1.0 + alpha * (target > 0).float()
    per_pixel = weight * (prob - target) ** 2
    per_sample = per_pixel.flatten(1).mean(dim=1)  # (B,)
    if valid is not None:
        per_sample = per_sample[valid]
        if per_sample.numel() == 0:
            return per_sample.sum()
    return per_sample.mean()
