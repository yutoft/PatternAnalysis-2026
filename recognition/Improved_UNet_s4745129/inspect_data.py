from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np

from dataset import HipMRIDataset, NUM_CLASSES


def main():
    dataset = HipMRIDataset(split="train")

    # Two training examples, including middle example from audit.
    indices = [0, len(dataset) // 2]
    cmap = plt.get_cmap("tab10", NUM_CLASSES)
    norm = BoundaryNorm(np.arange(NUM_CLASSES + 1) - 0.5, NUM_CLASSES)

    fig, axes = plt.subplots(2, 3, figsize=(10, 9))

    for row, index in enumerate(indices):
        image, mask = dataset[index]
        image = image.squeeze(0).numpy()
        mask = mask.numpy()

        # Transpose both arrays identically for display only.
        axes[row, 0].imshow(image.T, cmap="gray", origin="lower")
        axes[row, 0].set_title("Normalised MRI")

        mask_plot = axes[row, 1].imshow(
            mask.T, cmap=cmap, norm=norm,
            origin="lower", interpolation="nearest"
        )
        axes[row, 1].set_title("Ground-truth labels")

        axes[row, 2].imshow(image.T, cmap="gray", origin="lower")
        axes[row, 2].imshow(
            np.ma.masked_where(mask.T == 0, mask.T),
            cmap=cmap, norm=norm, alpha=0.4,
            origin="lower", interpolation="nearest"
        )
        axes[row, 2].set_title("Overlay (label 0 hidden)")

        axes[row, 0].set_ylabel(dataset.images[index].name, fontsize=7)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.colorbar(
        mask_plot, ax=axes.ravel().tolist(),
        ticks=range(NUM_CLASSES), label="Label ID", shrink=0.6
    )

    output = Path("results/data_overlay.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()