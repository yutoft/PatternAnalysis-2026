from pathlib import Path
from itertools import combinations

import nibabel as nib
import numpy as np


ROOT = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
SPLITS = ["train", "validate", "test"]


def main():
    patient_ids = {}
    image_files = {}

    # Check file pairing and patient separation across all filenames.
    for split in SPLITS:
        images = sorted(
            (ROOT / f"keras_slices_{split}").glob("*.nii.gz")
        )
        masks = {
            p.name
            for p in (ROOT / f"keras_slices_seg_{split}").glob("*.nii.gz")
        }

        if not images:
            raise RuntimeError(f"No images found for {split}")

        image_files[split] = images
        patient_ids[split] = {
            p.name.split("_")[1] for p in images
        }
        expected_masks = {
            p.name.replace("case_", "seg_", 1) for p in images
        }

        print(f"\n=== {split}: file audit ===")
        print(f"Images: {len(images)}; masks: {len(masks)}")
        print(f"Patients ({len(patient_ids[split])}): "
              f"{sorted(patient_ids[split])}")
        print(f"Missing masks: {len(expected_masks - masks)}")
        print(f"Masks without images: {len(masks - expected_masks)}")

    for a, b in combinations(SPLITS, 2):
        overlap = patient_ids[a] & patient_ids[b]
        print(f"\nPatient overlap {a}/{b}: {sorted(overlap)}")

    # Inspect three pairs per split using the same selection as before.
    for split in SPLITS:
        images = image_files[split]
        indices = sorted({0, len(images) // 2, len(images) - 1})

        print(f"\n=== {split}: sampled image-mask checks ===")
        for index in indices:
            image_path = images[index]
            mask_path = (
                ROOT / f"keras_slices_seg_{split}"
                / image_path.name.replace("case_", "seg_", 1)
            )

            image = nib.load(image_path)
            mask = nib.load(mask_path)
            x = image.get_fdata()
            y = mask.get_fdata()
            labels, counts = np.unique(y, return_counts=True)

            print(f"\nFile: {image_path.name}")
            print(f"Image shape: {x.shape}; mask shape: {y.shape}")
            print(f"Image range: {x.min():.3f} to {x.max():.3f}")
            print(f"All values finite: "
                  f"{np.isfinite(x).all() and np.isfinite(y).all()}")
            print(f"Mask labels/counts: "
                  f"{dict(zip(labels.tolist(), counts.tolist()))}")
            print(f"Shapes match: {x.shape == y.shape}")
            print(f"Affines match: "
                  f"{np.allclose(image.affine, mask.affine)}")
            print(f"Voxel spacing: {image.header.get_zooms()}")


if __name__ == "__main__":
    main()