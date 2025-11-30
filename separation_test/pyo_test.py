# tests/test_engine.py
from separation.pyo_io import  AudioRingBuffer


# -------- AudioRingBuffer tests --------

def test_ringbuffer_push_pop_order():
    buf = AudioRingBuffer(max_blocks=2)
    buf.push("a")
    buf.push("b")
    assert buf.pop() == "a"
    assert buf.pop() == "b"
    assert buf.pop() is None


def test_ringbuffer_overflow_drops_oldest():
    buf = AudioRingBuffer(max_blocks=2)
    buf.push("a")
    buf.push("b")
    buf.push("c")  # 'a' should be dropped
    assert buf.pop() == "b"
    assert buf.pop() == "c"
    assert buf.pop() is None