"""
TRACE — 逆モデル用 Dataset

../data/backward_pairs.npz (観測地点ごとのX_gaussian, Y, split) と
../data/current_field.npz (forward/dataset.pyと同じ、全site共通のu, v) を読み込み、
指定したsplit("train"/"val"/"test")のサンプルだけを取り出す。

入力は3チャンネル: [観測点ガウシアン, 平均u, 平均v] (forwardと全く同じ形式)。
出力は責任マップ(128,128, 合計1)。forwardのvalidフラグに相当するものは無い
(backward_pairs.npzの観測セルは構成上すべてY.sum()==1なので、常にTrueを返す
だけのダミーにしてforward/train.pyの訓練ループをそのまま使い回せるようにしている)。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_NPZ = DATA_DIR / "backward_pairs.npz"
DEFAULT_CURRENT_NPZ = DATA_DIR / "current_field.npz"


def load_coastal_mask(current_npz_path=DEFAULT_CURRENT_NPZ):
    """originは必ず沿岸(陸に接した海セル)のはず(release_site.pyがland maskから
    沿岸セルだけを抽出して300地点を作っている)。このmaskをmasked softmaxに使うと、
    モデルが非沿岸セルに確率を漏らせなくなる(predict_point.pyで発見した問題への対応)。
    戻り値は(128,128)のbool tensor、Trueが「確率を置いてよいセル」。"""
    from scipy.ndimage import binary_dilation

    c = np.load(current_npz_path)
    land = c["land_mask"]
    coastal = binary_dilation(land) & ~land
    return torch.from_numpy(coastal)


class BackwardDataset(Dataset):
    def __init__(self, split, npz_path=DEFAULT_NPZ, current_npz_path=DEFAULT_CURRENT_NPZ):
        d = np.load(npz_path)
        mask = d["split"] == split
        if not mask.any():
            raise ValueError(f"split={split!r} に該当するサンプルが0件です")

        self.X_gaussian = d["X_gaussian"][mask]
        self.Y = d["Y"][mask]
        self.n_origins = d["n_origins"][mask]
        self.row = d["row"][mask]
        self.col = d["col"][mask]

        if "release_date" in d.files:
            # 放出日ごとにcurrent_field_{date}.npzを読み、日付文字列をキーにして持つ
            # (forward/dataset.pyと同じ理由。詳細はそちらのdocstring参照)
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
            self.release_date = None
            c = np.load(current_npz_path)
            self.u = torch.from_numpy(c["u"]).float()
            self.v = torch.from_numpy(c["v"]).float()

    def __len__(self):
        return len(self.X_gaussian)

    def __getitem__(self, idx):
        gaussian = torch.from_numpy(self.X_gaussian[idx]).unsqueeze(0).float()
        if self.release_date is not None:
            u, v = self.current_by_date[self.release_date[idx]]
        else:
            u, v = self.u, self.v
        x = torch.cat([gaussian, u.unsqueeze(0), v.unsqueeze(0)], dim=0)  # (3,128,128)
        y = torch.from_numpy(self.Y[idx]).unsqueeze(0).float()
        valid = torch.tensor(True)  # forward/train.pyとの互換用ダミー(常にTrue)
        return x, y, valid
