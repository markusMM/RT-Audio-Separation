from typing import NoReturn
from separation.separator import Separator
from threading import Thread
from separation.pyo_io import AudioRingBuffer, PyoIO

# ----------------- realtime engine -----------------

class RealtimeSeparatorEngine:
    def __init__(self, stem: str = "vocals") -> None:
        self.separator = Separator()
        self.in_buffer = AudioRingBuffer()
        self.out_buffer = AudioRingBuffer()
        self.io = PyoIO(self.separator.sr, self.separator.block_size,
                        self.in_buffer, self.out_buffer)
        self.stem = stem
        self.worker = Thread(target=self._worker_loop, daemon=True)

    def _worker_loop(self) -> NoReturn:
        while True:
            block = self.in_buffer.pop()
            if block is None:
                continue
            sep_block = self.separator.separate_block(block, stem=self.stem)
            self.out_buffer.push(sep_block)

    def run(self) -> None:
        self.worker.start()
        self.io.start()
        try:
            while True:
                self.io.playback_step()
        except KeyboardInterrupt:
            pass

# ----------------- main -----------------

if __name__ == "__main__":
    engine = RealtimeSeparatorEngine(stem="vocals")
    engine.run()
