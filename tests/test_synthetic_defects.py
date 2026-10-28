import unittest

from PIL import Image

from src.synthetic_defects import apply_defect, generate_defect_mask


class SyntheticDefectTests(unittest.TestCase):
    def test_generate_defect_mask_shape_and_nonzero_pixels(self) -> None:
        mask = generate_defect_mask((128, 128), defect_type="blob")
        self.assertEqual(mask.shape, (128, 128))
        self.assertGreater(mask.sum(), 0)

    def test_apply_defect_increases_pixel_intensity(self) -> None:
        image = Image.new("RGB", (128, 128), color=(10, 10, 10))
        synthetic = apply_defect(image, defect_type="blob", intensity=0.8)
        self.assertEqual(synthetic.size, image.size)
        self.assertEqual(synthetic.getbands(), image.getbands())
        self.assertIsNotNone(synthetic.getbbox())


if __name__ == "__main__":
    unittest.main()
