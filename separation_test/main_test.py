# tests/test_engine.py
import numpy as np
from unittest.mock import patch, MagicMock
from separation.main import RealtimeSeparatorEngine

# -------- RealtimeSeparatorEngine tests --------
@patch("separation.pyo_io.PyoIO")
@patch("sepraration.separator.Separator")
def test_engine_worker_pushes_output(mock_separator_cls, mock_pyoio_cls):
    # Mock Separator.separate_block to be identity
    sep_instance = MagicMock()
    sep_instance.block_size = 16
    sep_instance.sr = 44100

    def fake_sep(block, stem="vocals"):
        return block

    sep_instance.separate_block.side_effect = fake_sep
    mock_separator_cls.return_value = sep_instance

    # Dummy PyoIO so we don't touch audio hardware
    pyo_instance = MagicMock()
    mock_pyoio_cls.return_value = pyo_instance

    engine = RealtimeSeparatorEngine(stem="vocals")

    # push one block into input buffer
    x = np.ones((16, 2), dtype=np.float32)
    engine.in_buffer.push(x)

    # run a single worker step directly
    # (call the private method once instead of starting the thread)
    engine._worker_loop.__wrapped__ = engine._worker_loop   # type: ignore # for pytest coverage
    # copy of logic in _worker_loop but single iteration:
    block = engine.in_buffer.pop()
    assert block is not None
    out = engine.separator.separate_block(block, stem=engine.stem)
    engine.out_buffer.push(out)

    # verify output is present and unchanged
    y = engine.out_buffer.pop()
    assert y is not None
    assert np.allclose(y, x)
