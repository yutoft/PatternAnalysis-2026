from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DATA_ROOT = Path(
    "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
)
NUM_CLASSES = 6


class HipMRIDataset(Dataset):
    """Load supplied HipMRI 2D splits without changing patient assignments."""

    def __init__(self, root=DATA_ROOT, split="train"):
        if split not in {"train", "validate", "test"}:
            raise ValueError("split must be train, validate or test")

        root = Path(root)
        self.images = sorted(
            (root / f"keras_slices_{split}").glob("case_*.nii.gz")
        )
        self.mask_dir = root / f"keras_slices_seg_{split}"

        if not self.images:
            raise RuntimeError(f"No images found for split: {split}")

        for image_path in self.images:
            mask_path = self.mask_dir / image_path.name.replace(
                "case_", "seg_", 1
            )
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name.replace(
            "case_", "seg_", 1
        )

        image_file = nib.load(image_path)
        mask_file = nib.load(mask_path)
        image = image_file.get_fdata(dtype=np.float32)
        mask = mask_file.get_fdata(dtype=np.float32)

        # Check the pair before preprocessing.
        if image.ndim != 2 or image.shape != mask.shape:
            raise ValueError(f"Invalid image/mask shapes: {image_path.name}")

        if not np.allclose(image_file.affine, mask_file.affine):
            raise ValueError(f"Affine mismatch: {image_path.name}")

        if not np.isfinite(image).all() or not np.isfinite(mask).all():
            raise ValueError(f"Non-finite values: {image_path.name}")

        if not np.equal(mask, np.round(mask)).all():
            raise ValueError(f"Non-integer mask labels: {mask_path.name}")

        if mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(f"Unexpected mask labels: {mask_path.name}")

        # Per-image z-score normalisation; constant images become zero.
        image = (image - image.mean()) / max(float(image.std()), 1e-8)

        # Image: [channels, height, width]. Mask: [height, width].
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask


def masks_to_one_hot(masks):
    """Convert batched integer masks [B,H,W] to [B,6,H,W]."""
    return F.one_hot(
        masks, num_classes=NUM_CLASSES
    ).permute(0, 3, 1, 2).float()


if __name__ == "__main__":
    dataset = HipMRIDataset(split="train")
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0
    )

    images, masks = next(iter(loader))
    one_hot_masks = masks_to_one_hot(masks)

    assert images.shape == (4, 1, 256, 128)
    assert masks.shape == (4, 256, 128)
    assert torch.all(one_hot_masks.sum(dim=1) == 1)

    print("Training samples:", len(dataset))
    print("Image batch:", images.shape, images.dtype)
    print("Mask batch:", masks.shape, masks.dtype)
    print("One-hot masks:", one_hot_masks.shape)
    print("Labels in this batch:", torch.unique(masks).tolist())
    print("Batch loading checks passed.")