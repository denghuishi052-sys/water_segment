import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "00_prepare_gf_floodnet_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_gf_floodnet_dataset", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareGFFloodNetDatasetTests(unittest.TestCase):
    def test_black_label_pixels_are_water_foreground(self):
        module = load_script()
        label = np.array([[0, 255, 127], [128, 200, 10]], dtype=np.uint8)

        mask = module.gf_label_to_mask(label, threshold=128)

        expected = np.array([[255, 0, 255], [0, 0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_five_channel_uint16_image_converts_to_rgb_uint8(self):
        module = load_script()
        image = np.zeros((2, 2, 5), dtype=np.uint16)
        image[..., 4] = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        image[..., 2] = np.array([[50, 60], [70, 80]], dtype=np.uint16)
        image[..., 0] = np.array([[5, 5], [5, 5]], dtype=np.uint16)

        rgb = module.to_rgb_uint8(image, rgb_channels=(4, 2, 0), stretch="percentile")

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreater(int(rgb[..., 0].max()), int(rgb[..., 0].min()))
        self.assertGreater(int(rgb[..., 1].max()), int(rgb[..., 1].min()))
        self.assertTrue(np.all(rgb[..., 2] == 0))


if __name__ == "__main__":
    unittest.main()
