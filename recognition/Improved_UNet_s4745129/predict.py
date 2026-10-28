"""Compare HipMRI checkpoints using the existing dataset and model definitions.

Distances/spatial outputs refer to resized 256 x 128 slices. Label names are
deliberately left unassigned until the original dataset label key is verified.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def confusion_matrix(target, prediction, num_classes=6):
    """Rows are true labels; columns are predicted labels."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError("Prediction and target shapes differ")
    for values in (target, prediction):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Labels must be integers")
        if values.size == 0 or values.min() < 0 or values.max() >= num_classes:
            raise ValueError("Empty masks or labels outside the expected range")
    counts = np.bincount(
        (num_classes * target.astype(np.int64) + prediction).ravel(),
        minlength=num_classes ** 2,
    )
    return counts.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion):
    """Compute pooled per-class overlap; absent in both masks is undefined."""
    confusion = np.asarray(confusion, dtype=np.float64)
    intersection = confusion.diagonal()
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    union = denominator - intersection
    dice = np.divide(
        2 * intersection, denominator,
        out=np.full_like(denominator, np.nan), where=denominator > 0,
    )
    iou = np.divide(
        intersection, union,
        out=np.full_like(union, np.nan), where=union > 0,
    )
    return dice, iou


def defined_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None


def json_values(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path, expected_model, device):
    import torch
    from modules import UNET, ImprovedUNet

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if config.get("model") != expected_model:
        raise ValueError(f"{path}: expected config.model={expected_model!r}")
    if config.get("num_classes") != 6:
        raise ValueError(f"{path}: expected six output classes")
    model_class = UNET if expected_model == "baseline" else ImprovedUNet
    model = model_class(in_channels=1, out_channels=6)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    metadata = {
        "checkpoint": str(path.resolve()),
        "sha256": file_hash(path),
        "epoch": checkpoint.get("epoch"),
        "checkpoint_validation_mean_dice": checkpoint.get("mean_dice"),
        "config": config,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, metadata


def save_figure(case, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    colors = ["#172133", "#39b8a4", "#e9bc55", "#a18bdb", "#e17670", "#55a4d8"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(7) - 0.5, cmap.N)
    fig, axes = plt.subplots(1, 4, figsize=(10, 6))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.15, top=0.84, wspace=0.12)
    axes[0].imshow(case["image"], cmap="gray", interpolation="nearest")
    axes[0].set_title("Normalised MRI", fontsize=11)
    axes[1].imshow(case["target"], cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Ground truth", fontsize=11)
    for axis, name in zip(axes[2:], ("baseline", "improved")):
        axis.imshow(case[name], cmap=cmap, norm=norm, interpolation="nearest")
        score = case[f"{name}_score"]
        score_text = "undefined" if score is None else f"{score:.3f}"
        axis.set_title(f"{name.capitalize()}\nDice (labels 1–5): {score_text}", fontsize=10)
    for axis in axes:
        axis.axis("off")
    fig.legend(
        handles=[Patch(facecolor=color, label=f"Label {i}") for i, color in enumerate(colors)],
        loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=6, frameon=False, fontsize=9,
    )
    fig.suptitle(f"{case['filename']}\nResized pixel grid | dataset index {case['index']}", fontsize=11)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--split", choices=["validate", "test"], default="validate")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Partial-data check only; omit for full evaluation")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save all predictions as uint8 .npy arrays on the resized grid")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or args.examples < 0
            or (args.max_batches is not None and args.max_batches < 1)):
        parser.error("Invalid batch size, workers, examples or max-batches")
    for checkpoint in (args.baseline, args.improved):
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")

    import torch
    from itertools import islice
    from torch.utils.data import DataLoader
    from dataset import HipMRIDataset, NUM_CLASSES

    if NUM_CLASSES != 6:
        raise ValueError("This script expects the supplied six-class dataset")
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(selected_device)
    dataset_args = {"split": args.split}
    if args.root is not None:
        dataset_args["root"] = args.root
    dataset = HipMRIDataset(**dataset_args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    models, metadata = {}, {}
    for name in ("baseline", "improved"):
        models[name], metadata[name] = load_model(getattr(args, name), name, device)

    output = args.output or (
        Path("results/comparison") / f"{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty; choose a new folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        for name in models:
            (output / "masks" / name).mkdir(parents=True, exist_ok=True)
    totals = {name: np.zeros((6, 6), dtype=np.int64) for name in models}
    candidates = []
    processed = 0
    fields = ["index", "filename", "model", "mean_dice_all_labels", "mean_dice_labels_1_to_5"]
    fields += [f"dice_label_{i}" for i in range(6)] + [f"iou_label_{i}" for i in range(6)]
    batches = loader if args.max_batches is None else islice(loader, args.max_batches)
    print(f"Evaluating {args.split} on {device}; inference uses float32", flush=True)
    with (output / "per_slice.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for images, targets in batches:
                images_device = images.to(device, non_blocking=True)
                predictions = {}
                for name, model in models.items():
                    logits = model(images_device)
                    if tuple(logits.shape) != (images.shape[0], 6, *targets.shape[1:]):
                        raise ValueError(f"Unexpected output shape from {name}: {logits.shape}")
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f"Non-finite logits from {name}")
                    predictions[name] = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                    del logits
                for offset in range(images.shape[0]):
                    index = processed + offset
                    filename = dataset.images[index].name
                    target = targets[offset].numpy()
                    scores = {}
                    for name in models:
                        prediction = predictions[name][offset]
                        confusion = confusion_matrix(target, prediction)
                        totals[name] += confusion
                        dice, iou = metrics_from_confusion(confusion)
                        scores[name] = defined_mean(dice[1:])
                        row = {"index": index, "filename": filename, "model": name,
                               "mean_dice_all_labels": defined_mean(dice),
                               "mean_dice_labels_1_to_5": scores[name]}
                        row.update({f"dice_label_{i}": value for i, value in enumerate(json_values(dice))})
                        row.update({f"iou_label_{i}": value for i, value in enumerate(json_values(iou))})
                        writer.writerow(row)
                        if args.save_masks:
                            stem = filename.removesuffix(".nii.gz")
                            np.save(output / "masks" / name / f"{stem}.npy", prediction)

                    if args.examples:
                        pair_score = defined_mean([
                            np.nan if scores[name] is None else scores[name] for name in models
                        ])
                        rank_score = float("inf") if pair_score is None else pair_score
                        candidate = {
                            "index": index, "filename": filename, "rank_score": rank_score,
                            "image": images[offset, 0].numpy().copy(), "target": target.copy(),
                            **{name: predictions[name][offset].copy() for name in models},
                            **{f"{name}_score": scores[name] for name in models},
                        }
                        candidates.append(candidate)
                        candidates.sort(key=lambda case: (case["rank_score"], case["index"]))
                        candidates = candidates[:args.examples]
                processed += images.shape[0]
                if processed == images.shape[0] or processed % 100 < images.shape[0]:
                    print(f"Evaluated {processed}/{len(dataset)} slices", flush=True)
    report = {
        "split": args.split, "dataset_slices": len(dataset), "evaluated_slices": processed,
        "partial_data": processed != len(dataset),
        "smoke_checkpoint_used": any(item["config"].get("smoke", True) for item in metadata.values()),
        "device": str(device), "inference_precision": "float32",
        "batch_size": args.batch_size, "num_workers": args.workers,
        "preprocessing": "Existing HipMRIDataset: per-slice z-score; bilinear images and nearest masks to 256x128",
        "labels": "Numeric IDs only; anatomical names and background ID are not verified",
        "metric_definition": "Per-class Dice/IoU from pooled pixels across evaluated resized slices; absent in both is null",
        "figure_selection": "Lowest mean of both models' per-slice mean Dice over labels 1-5; label 0 excluded by explicit convention",
        "models": {}, "figures": [],
    }
    for name in models:
        dice, iou = metrics_from_confusion(totals[name])
        report["models"][name] = {
            **metadata[name], "confusion_matrix": totals[name].tolist(),
            "dice": json_values(dice), "iou": json_values(iou),
            "mean_dice_all_labels": defined_mean(dice),
            "mean_dice_labels_1_to_5": defined_mean(dice[1:]),
            "mean_iou_all_labels": defined_mean(iou),
            "mean_iou_labels_1_to_5": defined_mean(iou[1:]),
        }
        print(f"\n{name} pooled metrics:", flush=True)
        for label in range(6):
            print(f"  Label {label}: Dice={dice[label]:.4f}; IoU={iou[label]:.4f}")
    if candidates:
        (output / "figures").mkdir()
    for rank, case in enumerate(candidates, start=1):
        relative_path = f"figures/lowest_{rank:02d}_index_{case['index']}.png"
        save_figure(case, output / relative_path)
        report["figures"].append({
            "path": relative_path, "index": case["index"], "filename": case["filename"],
            "baseline_mean_dice_labels_1_to_5": case["baseline_score"],
            "improved_mean_dice_labels_1_to_5": case["improved_score"],
        })
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["partial_data"] or report["smoke_checkpoint_used"]:
        print("\nCHECK ONLY: partial data or smoke checkpoint used; these are not final results.")
    print(f"\nSaved metrics.json, per_slice.csv and {len(candidates)} figures to {output}", flush=True)
if __name__ == "__main__":
    main()