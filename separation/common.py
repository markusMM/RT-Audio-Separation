import torch

# ----------------- config -----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SEC = 1.0      # processing block length (s)
