# ----------------- buffers -----------------

from collections import deque
from threading import Lock
from typing import Any

import numpy as np
from pyo import Input, Pan, Server, Sig


class AudioRingBuffer:
    def __init__(self, max_blocks: int = 32) -> None:
        self.buf = deque(maxlen=max_blocks)
        self.lock = Lock()

    def push(self, block: Any) -> None:
        with self.lock:
            self.buf.append(block)

    def pop(self) -> (None | Any):
        with self.lock:
            if not self.buf:
                return None
            return self.buf.popleft()


# ----------------- Pyo IO wrapper -----------------

class PyoIO:
    def __init__(
        self, 
        sr: int, 
        block_size: int, 
        in_buffer: AudioRingBuffer,
        out_buffer: AudioRingBuffer
    ) -> None:
        self.sr = sr
        self.block_size = block_size
        self.in_buffer = in_buffer
        self.out_buffer = out_buffer

        self.server = Server(
            sr=self.sr, 
            nchnls=2,
            buffersize=512, 
            duplex=1
        ).boot()
        self.mic = Input()
        self.play_sig = Sig(np.zeros(self.block_size, dtype=np.float32))
        self.play_pan = Pan(self.play_sig, outs=2).out()

        self.server.setCallback(self._record_callback)

    # ---- callbacks ----

    def _record_callback(self):
        data = self.mic.get(all=True)
        nchnls = self.server.getNchnls()
        frames = len(data) // nchnls
        block = np.reshape(
            np.array(data, dtype=np.float32),
            (frames, nchnls)
        )
        self.in_buffer.push(block)

    def playback_step(self) -> None:
        block = self.out_buffer.pop()
        if block is None:
            return
        # mix to mono
        if block.ndim == 2 and block.shape[1] > 1:
            block = block.mean(axis=1)
        if len(block) < self.block_size:
            pad = np.zeros(self.block_size - len(block), dtype=np.float32)
            block = np.concatenate([block, pad])
        else:
            block = block[:self.block_size]
        self.play_sig.value = block

    def start(self):
        self.server.start()