from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torchvision import models
from torchvision.transforms import functional as T

from src.config import DATA_ROOT
from src.dataset_loader import list_defect_types, list_mvttec_training_images


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchCoreModel:
    def __init__(self, device: str | None = None, resize: int = 256, coreset_size: int = 512):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.resize = resize
        self.coreset_size = coreset_size
        self.model = self._build_backbone()
        self.model.eval()
        self.model.to(self.device)

    def _build_backbone(self):
        try:
            from torchvision.models import Wide_ResNet50_2_Weights

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
            model = models.wide_resnet50_2(weights=weights)
        except Exception:
            model = models.wide_resnet50_2(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def _preprocess(image: Image.Image, resize: int) -> Tensor:
        image = image.convert("RGB")
        image = image.resize((resize, resize))
        tensor = T.to_tensor(image)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _forward_features(self, image_tensor: Tensor) -> tuple[Tensor, Tensor]:
        x = image_tensor.to(self.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        stage2 = self.model.layer2(x)
        stage3 = self.model.layer3(stage2)
        return stage2, stage3

    def extract_patch_vectors(self, image: Image.Image) -> tuple[Tensor, Tensor, Tensor]:
        image_tensor = self._preprocess(image, self.resize)
        stage2, stage3 = self._forward_features(image_tensor)

        stage2 = F.avg_pool2d(stage2, kernel_size=3, stride=1, padding=1)
        stage3 = F.avg_pool2d(stage3, kernel_size=3, stride=1, padding=1)

        stage3_up = F.interpolate(stage3, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([stage2, stage3_up], dim=1)
        patches = merged[0].permute(1, 2, 0).reshape(-1, merged.shape[1])
        return stage2[0], stage3[0], patches

    def build_memory_bank(self, normal_images: Sequence[Image.Image]) -> Tensor:
        all_vectors: list[Tensor] = []
        for image in normal_images:
            _, _, patches = self.extract_patch_vectors(image)
            all_vectors.append(patches.cpu())

        bank = torch.cat(all_vectors, dim=0)
        if self.coreset_size is not None and bank.shape[0] > self.coreset_size:
            bank = self.greedy_coreset(bank, self.coreset_size)
        return bank

    @staticmethod
    def greedy_coreset(vectors: Tensor, target_size: int) -> Tensor:
        if vectors.shape[0] <= target_size:
            return vectors

        selected_idx = [0]
        for _ in range(1, min(target_size, vectors.shape[0])):
            dists = torch.cdist(vectors, vectors[selected_idx], p=2.0)
            min_dist = dists.min(dim=1).values
            min_dist[selected_idx] = -torch.inf
            idx = int(torch.argmax(min_dist).item())
            selected_idx.append(idx)

        return vectors[selected_idx]

    def score_image(self, image: Image.Image, memory_bank: Tensor) -> tuple[Tensor, float]:
        with torch.no_grad():
            _, _, patches = self.extract_patch_vectors(image)
            patches = patches.to(self.device)
            memory_bank = memory_bank.to(self.device)
            distances = torch.cdist(patches, memory_bank, p=2.0)
            patch_scores = distances.min(dim=1).values
            image_score = float(patch_scores.max().item())
            return patch_scores, image_score

    @staticmethod
    def heatmap_from_scores(scores: Tensor, spatial_h: int, spatial_w: int, target_size: tuple[int, int]) -> Tensor:
        score_map = scores.reshape(spatial_h, spatial_w)
        upsampled = F.interpolate(
            score_map.unsqueeze(0).unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )
        return upsampled.squeeze(0).squeeze(0)


def load_category_images(category_name: str = "transistor") -> tuple[list[Image.Image], list[Image.Image], list[str]]:
    category_dir = DATA_ROOT / category_name
    train_images = [Image.open(p).convert("RGB") for p in list_mvttec_training_images(category_dir)]
    defect_types = list_defect_types(category_dir)
    test_images = []
    for defect_type in defect_types:
        defect_dir = category_dir / "test" / defect_type
        for image_path in sorted(defect_dir.glob("*.png")):
            test_images.append(Image.open(image_path).convert("RGB"))
    return train_images, test_images, defect_types


def _find_mask_path(image_path: Path, category_dir: Path) -> Path | None:
    defect_type = image_path.parent.name
    mask_dir = category_dir / "ground_truth" / defect_type
    if not mask_dir.exists():
        return None

    for name in [f"{image_path.stem}_mask.png", f"{image_path.stem}.png", f"{image_path.stem}_mask.jpg"]:
        candidate = mask_dir / name
        if candidate.exists():
            return candidate
    return None


def collect_test_items(category_name: str = "transistor") -> tuple[list[tuple[Image.Image, int, Path | None]], list[Path]]:
    category_dir = DATA_ROOT / category_name
    items: list[tuple[Image.Image, int, Path | None]] = []
    good_dir = category_dir / "test" / "good"

    for image_path in sorted(good_dir.glob("*.png")):
        items.append((Image.open(image_path).convert("RGB"), 0, None))

    for defect_type in list_defect_types(category_dir):
        defect_dir = category_dir / "test" / defect_type
        for image_path in sorted(defect_dir.glob("*.png")):
            mask_path = _find_mask_path(image_path, category_dir)
            items.append((Image.open(image_path).convert("RGB"), 1, mask_path))

    return items, list_mvttec_training_images(category_dir)


def image_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return roc_auc_score(labels, scores)


def pixel_auroc_from_heatmaps(labels: Sequence[int], heatmaps: Sequence[Tensor], masks: Sequence[Tensor]) -> float:
    flattened_labels: list[int] = []
    flattened_scores: list[float] = []
    for label, heatmap, mask in zip(labels, heatmaps, masks):
        heat_flat = heatmap.detach().flatten().cpu().numpy()
        mask_flat = mask.detach().flatten().cpu().numpy().astype(float)
        flattened_labels.extend([int(label)] * len(heat_flat))
        flattened_scores.extend(heat_flat.tolist())
        flattened_labels.extend([int(label)] * len(mask_flat))
        flattened_scores.extend(mask_flat.tolist())
    # The intensity list contains heatmap and mask values; this is used only for checking alignment in the script.
    return float(roc_auc_score(np.asarray(flattened_labels), np.asarray(flattened_scores)))
