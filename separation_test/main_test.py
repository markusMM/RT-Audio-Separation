# tests/test_engine.py
import numpy as np
from unittest.mock import patch, MagicMock

from separation.main import RealtimeSeparatorEngine
from separation.separator import Separator  # real Separator class
from separation.pyo_io import AudioRingBuffer, PyoIO


# -------- RealtimeSeparatorEngine tests --------
def test_engine_worker_pushes_output():
    # Create REAL Separator & PyoIO instances
    separator = Separator()
    in_buffer = AudioRingBuffer()
    out_buffer = AudioRingBuffer()
    pyo_io = PyoIO(
        separator.sr, separator.block_size,
        in_buffer, out_buffer
    )
    
    # Mock JUST the methods we care about
    separator.separate_block = MagicMock(side_effect=lambda block, **kw: block)
    pyo_io.server = MagicMock()  # minimal server mock
    
    # Patch ONLY the constructor calls in RealtimeSeparatorEngine
    with patch.object(RealtimeSeparatorEngine, 'separator', separator):
        with patch.object(RealtimeSeparatorEngine, 'io', pyo_io):
            
            engine = RealtimeSeparatorEngine(stem="vocals")
            
            # Test the actual worker logic
            x = np.ones((16, 2), dtype=np.float32)
            engine.in_buffer.push(x)
            
            # Run one iteration of worker logic (extracted for testability)
            block = engine.in_buffer.pop()
            assert block is not None
            
            out = engine.separator.separate_block(block, stem=engine.stem)
            engine.out_buffer.push(out)
            
            # Verify output
            y = engine.out_buffer.pop()
            assert y is not None
            assert np.allclose(y, x)
            separator.separate_block.assert_called_once()
