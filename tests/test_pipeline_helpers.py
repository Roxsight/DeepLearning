import tempfile
import unittest
from pathlib import Path
from src.common import label_from_path, stable_id

class TestPipelineHelpers(unittest.TestCase):
    def test_fakeavceleb_category_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cases = {
                "RealVideo-RealAudio": 0,
                "FakeVideo-RealAudio": 1,
                "RealVideo-FakeAudio": 1,
                "FakeVideo-FakeAudio": 1,
            }
            for category, expected in cases.items():
                p = root / category / "person1" / "clip.mp4"
                p.parent.mkdir(parents=True)
                p.touch()
                self.assertEqual(label_from_path(p, root)[0], expected)

    def test_stable_id(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "RealVideo-RealAudio" / "x.mp4"
            p.parent.mkdir(parents=True)
            p.touch()
            self.assertEqual(stable_id(p, root), stable_id(p, root))

if __name__ == "__main__":
    unittest.main()
