# 逆モデル用 Loss


import importlib.util
from pathlib import Path

_forward_losses_path = Path(__file__).resolve().parent.parent / "forward" / "losses.py"
_spec = importlib.util.spec_from_file_location("forward_losses", _forward_losses_path)
_forward_losses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_forward_losses)

cross_entropy_loss = _forward_losses.cross_entropy_loss
to_prob_map = _forward_losses.to_prob_map
weighted_mse_loss = _forward_losses.weighted_mse_loss
