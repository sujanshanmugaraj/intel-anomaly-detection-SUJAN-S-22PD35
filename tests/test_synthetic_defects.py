import unittest

import numpy as np
import torch
from PIL import Image

from src.patchcore import PatchCoreModel, image_auroc
from src.synthetic_defects import apply_defect, generate_defect_mask


class SyntheticDefectTests(unittest.TestCase):
    def test_generate_defect_mask_shape_and_nonzero_pixels(self) -> None:
        mask = generate_defect_mask((128, 128), defect_type="blob")
        self.assertEqual(mask.shape, (128, 128))
        self.assertGreater(mask.sum(), 0)

    def test_apply_defect_changes_only_masked_pixels(self) -> None:
        image = Image.new("RGB", (128, 128), color=(10, 10, 10))
        synthetic, mask = apply_defect(image, defect_type="blob", intensity=0.8)
        self.assertEqual(synthetic.size, image.size)
        self.assertEqual(synthetic.getbands(), image.getbands())
        self.assertFalse(np.array_equal(np.asarray(synthetic), np.asarray(image)))

        generated = np.asarray(synthetic)
        original = np.asarray(image)
        changed = np.any(generated != original, axis=-1)
        self.assertTrue(np.any(changed))
        self.assertTrue(np.all(~changed | (mask > 0)))

    def test_greedy_coreset_toy_example(self) -> None:
        points = torch.tensor([[0.0], [1.0], [2.0], [10.0], [11.0], [20.0]], dtype=torch.float32)
        selected = PatchCoreModel.greedy_coreset(points, target_size=4)
        selected_set = set(int(x.item()) for x in selected.squeeze(1))
        self.assertEqual(selected_set, {0, 2, 10, 20})

    def test_hand_computed_auroc(self) -> None:
        labels = [1, 1, 0, 0, 0]
        scores = [0.9, 0.6, 0.7, 0.5, 0.1]
        self.assertAlmostEqual(image_auroc(labels, scores), 5.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
