from pathlib import Path
from guca.vis.png import save_png

# Minimal node/edge shapes the renderer understands
class _Node:
    def __init__(self, nid, state):
        self.id = nid
        self.state = state  # can be 'H','G','C' or any other label/number

class _Graph:
    def __init__(self):
        self._n = [
            _Node(0, "G"),
            _Node(1, "C"),
            _Node(2, "H"),
            _Node(3, "H"),
        ]
        # star + one extra edge to show label/color variety
        self._e = [(self._n[0], self._n[1]), (self._n[0], self._n[2]), (self._n[0], self._n[3])]

    def nodes(self):
        return self._n

    def edges(self):
        return self._e

def test_save_png_smoke(tmp_path):
    g = _Graph()
    out = tmp_path / "runs" / "demo" / "vis" / "step3.png"
    t = save_png(g, out)
    assert out.exists(), "PNG was not created"
    assert out.stat().st_size > 0, "PNG file is empty"
