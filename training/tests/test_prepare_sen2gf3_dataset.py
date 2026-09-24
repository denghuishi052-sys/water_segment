import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


def load_script():
    script = Path(__file__).resolve().parents[1] / "scripts" / "00_prepare_sen2gf3_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_sen2gf3_dataset", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareSen2GF3DatasetTests(unittest.TestCase):
    def test_to_rgb_uint8_uses_first_three_channels_by_default(self):
        module = load_script()
        image = np.zeros((2, 2, 4), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        image[..., 3] = 99

        rgb = module.to_rgb_uint8(image)

        self.assertEqual(rgb.shape, (2, 2, 3))
        np.testing.assert_array_equal(rgb[..., 0], np.full((2, 2), 10, dtype=np.uint8))
        np.testing.assert_array_equal(rgb[..., 1], np.full((2, 2), 20, dtype=np.uint8))
        np.testing.assert_array_equal(rgb[..., 2], np.full((2, 2), 30, dtype=np.uint8))

    def test_to_rgb_uint8_can_select_channels_and_percentile_stretch(self):
        module = load_script()
        image = np.zeros((2, 2, 4), dtype=np.uint16)
        image[..., 3] = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        image[..., 1] = np.array([[10, 20], [30, 40]], dtype=np.uint16)
        image[..., 0] = 5

        rgb = module.to_rgb_uint8(image, rgb_channels=(3, 1, 0), stretch="percentile")

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreater(int(rgb[..., 0].max()), int(rgb[..., 0].min()))
        self.assertGreater(int(rgb[..., 1].max()), int(rgb[..., 1].min()))
        self.assertTrue(np.all(rgb[..., 2] == 0))

    def test_build_flood_pseudo_rgb_uses_ndwi_and_inverted_sar_channels(self):
        module = load_script()
        b2 = np.zeros((2, 2), dtype=np.uint8)
        b3 = np.array([[80, 80], [20, 20]], dtype=np.uint8)
        b4 = np.zeros((2, 2), dtype=np.uint8)
        b8 = np.array([[20, 20], [80, 80]], dtype=np.uint8)
        hh = np.array([[1, 10], [100, 1000]], dtype=np.uint16)
        hv = np.array([[1000, 100], [10, 1]], dtype=np.uint16)

        rgb = module.build_flood_pseudo_rgb(b2, b3, b4, b8, hh, hv)

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreater(int(rgb[0, 0, 0]), int(rgb[1, 0, 0]))
        self.assertGreater(int(rgb[0, 0, 1]), int(rgb[1, 1, 1]))
        self.assertLess(int(rgb[0, 0, 2]), int(rgb[1, 1, 2]))

    def test_label_to_mask_writes_binary_255_mask(self):
        module = load_script()
        label = np.array([[0, 1, 2], [0, 0, 7]], dtype=np.uint8)

        mask = module.label_to_mask(label)

        expected = np.array([[0, 255, 255], [0, 0, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_extract_numeric_id_matches_sentinel_and_label_names(self):
        module = load_script()

        self.assertEqual(module.extract_numeric_id(Path("before_s2_1000.TIF")), "1000")
        self.assertEqual(module.extract_numeric_id(Path("label_1000.tif")), "1000")
        self.assertEqual(module.extract_numeric_id(Path("after_gf3_hh_1000.TIF")), "1000")

    def test_prepare_dataset_skips_existing_outputs_without_reading_source_tifs(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Sen2GF3Floods"
            (root / "sentinel2").mkdir(parents=True)
            (root / "label").mkdir(parents=True)
            (root / "sentinel2" / "before_s2_1.TIF").write_text("not a tif", encoding="utf-8")
            (root / "label" / "label_1.tif").write_text("not a tif", encoding="utf-8")

            output = Path(tmp) / "raw"
            (output / "images").mkdir(parents=True)
            (output / "masks").mkdir(parents=True)
            cv2.imwrite(str(output / "images" / "sen2gf3_00001.jpg"), np.zeros((2, 3, 3), dtype=np.uint8))
            cv2.imwrite(str(output / "masks" / "sen2gf3_00001.png"), np.array([[0, 255, 0], [255, 0, 0]], dtype=np.uint8))

            df = module.prepare_dataset(root, output, overwrite=False)

        self.assertEqual(len(df), 1)
        self.assertEqual(int(df.iloc[0]["mask_area"]), 2)


if __name__ == "__main__":
    unittest.main()
