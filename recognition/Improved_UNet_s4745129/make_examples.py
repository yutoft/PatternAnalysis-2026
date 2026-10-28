"""Render chosen slice indices using the same checkpoint/plot helpers as predict.py."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--split", choices=["validate", "test"], default="validate")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.indices) != len(set(args.indices)):
        parser.error("Supply each index only once")
    for checkpoint in (args.baseline, args.improved):
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder is not empty; choose a new folder")

    import numpy as np
    import torch
    from dataset import HipMRIDataset
    from predict import (
        confusion_matrix, defined_mean, json_values, load_model,
        metrics_from_confusion, save_figure,
    )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    dataset_args = {"split": args.split}
    if args.root is not None:
        dataset_args["root"] = args.root
    dataset = HipMRIDataset(**dataset_args)
    if any(index < 0 or index >= len(dataset) for index in args.indices):
        parser.error(f"Indices must be between 0 and {len(dataset) - 1}")
    models, checkpoints = {}, {}
    for name in ("baseline", "improved"):
        models[name], checkpoints[name] = load_model(getattr(args, name), name, device)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "split": args.split, "indices": args.indices, "inference_precision": "float32",
        "selection": "Explicit illustrative indices; not a random sample or a dataset-level evaluation",
        "labels": "Numeric IDs; anatomical names have not been verified",
        "checkpoints": checkpoints, "examples": [],
    }
    with torch.inference_mode():
        for index in args.indices:
            image, target = dataset[index]
            inputs = image.unsqueeze(0).to(device)
            filename = dataset.images[index].name
            case = {"index": index, "filename": filename,
                    "image": image[0].numpy(), "target": target.numpy()}
            item = {"index": index, "filename": filename, "models": {}}
            for name, model in models.items():
                logits = model(inputs)
                if tuple(logits.shape) != (1, 6, *target.shape):
                    raise ValueError(f"Unexpected output shape from {name}: {logits.shape}")
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite logits from {name}")
                prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                dice, iou = metrics_from_confusion(confusion_matrix(target.numpy(), prediction))
                case[name] = prediction
                case[f"{name}_score"] = defined_mean(dice[1:])
                item["models"][name] = {
                    "dice": json_values(dice), "iou": json_values(iou),
                    "mean_dice_labels_1_to_5": case[f"{name}_score"],
                }
                del logits
            item["figure"] = f"example_index_{index}.png"
            save_figure(case, args.output / item["figure"])
            report["examples"].append(item)
            print(f"Saved index {index}: {filename}", flush=True)
    (args.output / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(args.indices)} examples to {args.output}", flush=True)

if __name__ == "__main__":
    main()