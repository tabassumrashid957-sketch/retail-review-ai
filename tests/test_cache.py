import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from shoplifting.extract import valid_cache


class CacheTests(unittest.TestCase):
    def test_replaced_source_or_broken_timestamps_invalidate_cache(self):
        config = {"feature_dim": 512}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            values = {"features": np.zeros((1, 512), dtype=np.float32), "starts": np.array([0], dtype=np.float32), "ends": np.array([2], dtype=np.float32), "source_sha256": "original", "config": json.dumps(config, sort_keys=True)}
            np.savez(path, **values)
            self.assertTrue(valid_cache(path, [(0, 2)], config, "original"))
            self.assertFalse(valid_cache(path, [(0, 2)], config, "replacement"))
            self.assertFalse(valid_cache(path, [(1, 3)], config, "original"))
            del values["ends"]
            np.savez(path, **values)
            self.assertFalse(valid_cache(path, [(0, 2)], config, "original"))


if __name__ == "__main__":
    unittest.main()
