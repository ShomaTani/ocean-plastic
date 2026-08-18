
# 逆モデル(Attention U-Net)

import importlib.util
from pathlib import Path

_forward_main_path = Path(__file__).resolve().parent.parent / "forward" / "main.py"
_spec = importlib.util.spec_from_file_location("forward_main", _forward_main_path)
_forward_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_forward_main)

AttentionGate = _forward_main.AttentionGate
AttentionUNet = _forward_main.AttentionUNet
ConvBlock = _forward_main.ConvBlock
DownSample = _forward_main.DownSample
UpSample = _forward_main.UpSample
