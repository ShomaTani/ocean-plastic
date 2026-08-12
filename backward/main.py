"""
TRACE — 逆モデル(Attention U-Net)

アーキテクチャはforward/main.pyのAttentionUNetと完全に同一(「観測点→責任マップ」も
「放出点→漂着分布」も、同じ128x128の画像対画像変換タスクという構造は変わらない)。
別実装を持つと2箇所を同じように直し続ける羽目になるので、forward/main.pyから
そのままimportして使う。

このファイル自身も"main.py"という名前なので、素朴に`sys.path`へforward/を足して
`from main import ...`すると、Pythonが自分自身(backward/main.py)を"main"として
先に解決してしまい循環importになる。importlibでファイルパスを直接指定して、
"forward_main"という別名でモジュール登録することで名前衝突を回避している。
"""

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
