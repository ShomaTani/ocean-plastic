"""
TRACE — 順モデル用 Dataset

../data/train_pairs.npz (300地点分のX, Y, split, valid) を読み込み、
指定したsplit("train"/"val"/"test")のサンプルだけを取り出して
PyTorchが扱えるtensor形式に変換する。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_NPZ = Path(__file__).resolve().parent.parent / "data" / "train_pairs.npz"


class TraceDataset(Dataset):
    def __init__(self, split, npz_path=DEFAULT_NPZ):
        d = np.load(npz_path)
        mask = d["split"] == split
        if not mask.any():
            raise ValueError(f"split={split!r} に該当するサンプルが0件です")

        # (300,128,128) のうち、このsplitに属する行だけ抜き出して保持しておく
        self.X = d["X"][mask]
        self.Y = d["Y"][mask]
        self.valid = d["valid"][mask]
        self.site_id = d["site_id"][mask]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Conv2dは (C, H, W) の形を要求するので、channel次元(C=1)を先頭に足す
        x = torch.from_numpy(self.X[idx]).unsqueeze(0).float()
        y = torch.from_numpy(self.Y[idx]).unsqueeze(0).float()
        valid = torch.tensor(bool(self.valid[idx]))
        return x, y, valid
