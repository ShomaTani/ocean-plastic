"""
順モデル用 Loss

AttentionUNet(main.py)はロジット(生の値、活性化なし)を返す設計にしてある。
ここでまず to_prob_map() で「画像全体で合計1になる確率分布」に変換してから、
2種類のlossのどちらかで正解(Y, 合計1の確率分布)と比較する。
cross_entropy_loss, weighted_mse_loss 
"""

import torch


def to_prob_map(logits, mask=None):
    B, C, H, W = logits.shape
    flat = logits.view(B, -1)          # (B, H*W)
    if mask is not None:
        flat = flat.masked_fill(~mask.reshape(1, -1), float("-inf"))
    prob = torch.softmax(flat, dim=1)  # 合計1
    return prob.view(B, C, H, W)


def _apply_mask_to_target(target, mask, eps=1e-8):
    if mask is None:
        return target
    masked = target * mask.reshape(1, 1, *mask.shape).float()
    total = masked.flatten(1).sum(dim=1).clamp_min(eps).view(-1, 1, 1, 1)
    return masked / total


def cross_entropy_loss(logits, target, valid=None, eps=1e-8, mask=None):
    target = _apply_mask_to_target(target, mask, eps)
    prob = to_prob_map(logits, mask=mask)
    per_sample = -(target * torch.log(prob + eps)).flatten(1).sum(dim=1)  # (B,)
    if valid is not None:
        per_sample = per_sample[valid]
        if per_sample.numel() == 0:
            return per_sample.sum()  # このバッチにvalidなサンプルが1つもない
    return per_sample.mean()


def weighted_mse_loss(logits, target, alpha=5.0, valid=None, mask=None):
    target = _apply_mask_to_target(target, mask, eps=1e-8)
    prob = to_prob_map(logits, mask=mask)
    weight = 1.0 + alpha * (target > 0).float()
    per_pixel = weight * (prob - target) ** 2
    per_sample = per_pixel.flatten(1).mean(dim=1)  # (B,)
    if valid is not None:
        per_sample = per_sample[valid]
        if per_sample.numel() == 0:
            return per_sample.sum()
    return per_sample.mean()
