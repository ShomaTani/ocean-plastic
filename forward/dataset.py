"""
TRACE — 順モデル用 Dataset

../data/train_pairs.npz (300地点分のX, Y, split, valid) と
../data/current_field.npz (全origin共通の海流場u, v) を読み込み、
指定したsplit("train"/"val"/"test")のサンプルだけを取り出して
PyTorchが扱えるtensor形式に変換する。

入力は3チャンネル: [放出点ガウシアン, 平均u, 平均v]。
u, vはsite非依存(全サンプルで同じ2枚)なので、__getitem__ のたびに
ガウシアンチャンネルとstackして返す。

(speedチャンネル(√(u²+v²))も試したが、u,vから決定的に計算できる冗長な情報で
過学習を悪化させただけだったので不採用。current_field.npz自体にはspeedも
残してあるので、必要になったら再度使える)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_NPZ = DATA_DIR / "train_pairs.npz"
DEFAULT_CURRENT_NPZ = DATA_DIR / "current_field.npz"


def load_coastal_mask(dilation=3, current_npz_path=DEFAULT_CURRENT_NPZ):
    """漂着は陸に接触した場所でしか起きないはずだが、128x128の粗いグリッドだと
    小島や複雑な海岸線が陸マスクから欠落し、本物の漂着セルの一部が「非沿岸」に
    見えてしまう(dilation=1だと正解Yの48%が漏れる)。dilation回数を増やして
    許容範囲を緩めることで、本物の信号を削りすぎずに「あからさまに沖合すぎる
    予測」だけを抑える方向を狙う(bexp Aアプローチ、backward/dataset.pyの
    load_coastal_mask相当だが既定のdilationを緩めてある)。"""
    from scipy.ndimage import binary_dilation

    c = np.load(current_npz_path)
    land = c["land_mask"]
    coastal = binary_dilation(land, iterations=dilation) & ~land
    return torch.from_numpy(coastal)


class TraceDataset(Dataset):
    def __init__(self, split, npz_path=DEFAULT_NPZ, current_npz_path=DEFAULT_CURRENT_NPZ):
        d = np.load(npz_path)
        mask = d["split"] == split
        if not mask.any():
            raise ValueError(f"split={split!r} に該当するサンプルが0件です")

        # (300,128,128) のうち、このsplitに属する行だけ抜き出して保持しておく
        self.X = d["X"][mask]
        self.Y = d["Y"][mask]
        self.valid = d["valid"][mask]
        self.site_id = d["site_id"][mask]

        c = np.load(current_npz_path)
        # (128,128) が1枚ずつ。全site共通なので __init__ で1回だけ読めば十分
        self.u = torch.from_numpy(c["u"]).float()
        self.v = torch.from_numpy(c["v"]).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Conv2dは (C, H, W) の形を要求するので、channel次元(C=1)を先頭に足す
        gaussian = torch.from_numpy(self.X[idx]).unsqueeze(0).float()
        x = torch.cat([gaussian, self.u.unsqueeze(0), self.v.unsqueeze(0)], dim=0)  # (3,128,128)
        y = torch.from_numpy(self.Y[idx]).unsqueeze(0).float()
        valid = torch.tensor(bool(self.valid[idx]))
        return x, y, valid
