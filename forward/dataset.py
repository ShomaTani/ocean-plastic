"""
順モデル用 Dataset

../data/train_pairs.npz (500地点 x 5放出日分のX, Y, split, valid, release_date) と
../data/current_field_{release_date}.npz (放出日ごとの海流場u, v) を読み込み、
指定したsplit("train"/"val"/"test")のサンプルだけを取り出して
PyTorchが扱えるtensor形式に変換する。

入力は3チャンネル: [放出点ガウシアン, u, v]。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_NPZ = DATA_DIR / "train_pairs.npz"
DEFAULT_CURRENT_NPZ = DATA_DIR / "current_field.npz"


def load_coastal_mask(dilation=3, current_npz_path=DEFAULT_CURRENT_NPZ):
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

        # (n_rows,128,128) のうち、このsplitに属する行だけ抜き出して保持しておく
        self.X = d["X"][mask]
        self.Y = d["Y"][mask]
        self.valid = d["valid"][mask]
        self.site_id = d["site_id"][mask]

        if "release_date" in d.files:
            # 放出日ごとにcurrent_field_{date}.npzを読み、日付文字列をキーにして持つ
            self.release_date = d["release_date"][mask]
            current_dir = Path(current_npz_path).parent
            self.current_by_date = {}
            for date in np.unique(self.release_date):
                c = np.load(current_dir / f"current_field_{date}.npz")
                self.current_by_date[date] = (
                    torch.from_numpy(c["u"]).float(),
                    torch.from_numpy(c["v"]).float(),
                )
        else:
            # 旧フォーマット(季節拡張前): 全サンプル共通の1枚のみ
            self.release_date = None
            c = np.load(current_npz_path)
            self.u = torch.from_numpy(c["u"]).float()
            self.v = torch.from_numpy(c["v"]).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Conv2dは (C, H, W) の形を要求するので、channel次元(C=1)を先頭に足す
        gaussian = torch.from_numpy(self.X[idx]).unsqueeze(0).float()
        if self.release_date is not None:
            u, v = self.current_by_date[self.release_date[idx]]
        else:
            u, v = self.u, self.v
        x = torch.cat([gaussian, u.unsqueeze(0), v.unsqueeze(0)], dim=0)  # (3,128,128)
        y = torch.from_numpy(self.Y[idx]).unsqueeze(0).float()
        valid = torch.tensor(bool(self.valid[idx]))
        return x, y, valid
