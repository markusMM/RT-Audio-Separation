from typing import Any
import torch
import numpy as np
from separation.common import BLOCK_SEC, DEVICE

# ----------------- separation model wrapper -----------------
class Separator:
    def __init__(self, model_name: str = "htdemucs") -> None:
        self.model = torch.hub.load(
            "facebookresearch/demucs", model_name
        ).eval().to(DEVICE)  # type: ignore
        self.sr = self.model.samplerate
        self.block_size = int(BLOCK_SEC * self.sr)

    def separate_block(self, block_np: np.ndarray, stem: str = "vocals") -> Any:
        """
        block_np: (T, C) float32, T≈block_size at model.sr
        returns: (T, C_out) float32
        """
        # ensure correct length
        if block_np.shape[0] < self.block_size:
            pad = np.zeros((self.block_size - block_np.shape[0],
                            block_np.shape[1]), dtype=np.float32)
            block_np = np.vstack([block_np, pad])
        else:
            block_np = block_np[:self.block_size]

        wav = torch.from_numpy(block_np).float().t().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(wav)      # dict of stems
        stem_np = out[stem][0].cpu().numpy().T  # (T, C)
        return stem_np.astype(np.float32)
