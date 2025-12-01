import numpy as np
import torch
from unittest.mock import patch, MagicMock

from separation.separator import Separator  # Adjust import to your project layout


class DummyDemucs:
    def __init__(self, samplerate=44100):
        self.samplerate = samplerate

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, wav):
        # wav shape: (1, C, T)
        return {"vocals": wav}


@patch("torch.hub.load")
def test_separator_respects_block_size(mock_hub_load):
    mock_hub_load.return_value = DummyDemucs()

    sep = Separator()  # Uses mocked Demucs now
    T = sep.block_size
    C = 2

    # Create input shorter than block size (half length)
    x = np.random.randn(T // 2, C).astype(np.float32)

    out = sep.separate_block(x, stem="vocals")

    assert isinstance(out, np.ndarray)
    assert out.shape == (T, C)
