# tests/test_engine.py
import numpy as np
from unittest.mock import patch, MagicMock
from separation.separator import Separator
import types

# -------- Separator tests (mocked Demucs) --------
class DummyDemucs:
    """Minimal stand‑in for the real Demucs model."""
    def __init__(self, samplerate=44100):
        self.samplerate = samplerate

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, wav):
        # wav: (1, C, T) -> return dict with same shape
        # here we just pass input through as 'vocals'
        return {"vocals": wav}


@patch("src.engine.torch")
def test_separator_respects_block_size(mock_torch):
    # --- mock torch + hub.load so no real model is downloaded ---
    mock_torch.device = "cpu"
    mock_torch.cuda.is_available.return_value = False

    dummy_model = DummyDemucs()
    hub = types.SimpleNamespace(load=MagicMock(return_value=dummy_model))
    mock_torch.hub = hub

    # torch.tensor / from_numpy / no_grad context
    def from_numpy(x):
        m = MagicMock()
        m.float.return_value = m
        m.t.return_value = m
        m.unsqueeze.return_value = m
        m.to.return_value = m
        return m

    mock_torch.from_numpy.side_effect = from_numpy

    class DummyNoGrad:
        def __enter__(self, *a): return None
        def __exit__(self, *a): return False

    mock_torch.no_grad.return_value = DummyNoGrad()

    sep = Separator()  # uses mocked model
    T = sep.block_size
    C = 2

    # input shorter than block size -> should be padded to T
    x = np.random.randn(T // 2, C).astype(np.float32)
    out = sep.separate_block(x, stem="vocals")
    assert out.shape == (T, C)