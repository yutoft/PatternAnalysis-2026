"""Profile fixed HipMRI checkpoints and render class-specific FP/FN overlays.

Requires dataset.py, modules.py and predict.py in the same project folder.
Timing is batch-one, float32 forward + argmax, with CUDA events after warmup.
Per-scan/per-case times sum slice GPU times; they exclude I/O, preprocessing,
host transfers, model loading and plotting. This does not train either model.
"""

import argparse
import csv
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def scan_identifiers(filename):
    match = re.fullmatch(r"(case_\d+)_week_(\d+)_slice_(\d+)\.nii\.gz", filename)
    if match is None:
        raise ValueError(f"Unexpected slice filename: {filename}")
    case_id, week, slice_number = match.groups()
    return case_id, f"{case_id}_week_{week}", int(slice_number)


def binary_errors(target, prediction, label):
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError("Prediction/target shapes differ")
    return (prediction == label) & (target != label), (target == label) & (prediction != label)


def save_overlay(case, label, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    colors = ["#172133", "#39b8a4", "#e9bc55", "#a18bdb", "#e17670", "#55a4d8"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(7) - 0.5, cmap.N)
    fig, axes = plt.subplots(3, 2, figsize=(7, 13))
    fig.subplots_adjust(left=.035, right=.965, bottom=.075, top=.92, hspace=.19, wspace=.12)
    axes[0, 0].imshow(case["image"], cmap="gray", interpolation="nearest")
    axes[0, 0].set_title("Normalised MRI", fontsize=10)
    axes[0, 1].imshow(case["target"], cmap=cmap, norm=norm, interpolation="nearest")
    axes[0, 1].set_title("Ground truth · labels 0–5", fontsize=10)
    counts = {}
    for column, name in enumerate(("baseline", "improved")):
        prediction = case[name]
        fp, fn = binary_errors(case["target"], prediction, label)
        counts[name] = {"false_positive_pixels": int(fp.sum()), "false_negative_pixels": int(fn.sum())}
        axes[1, column].imshow(prediction, cmap=cmap, norm=norm, interpolation="nearest")
        axes[1, column].set_title(f"{name.capitalize()} prediction", fontsize=10)
        axes[2, column].imshow(case["image"], cmap="gray", interpolation="nearest")
        rgba = np.zeros((*fp.shape, 4), dtype=np.float32)
        rgba[fp] = [0.95, 0.15, 0.15, 0.85]
        rgba[fn] = [0.12, 0.48, 1.0, 0.85]
        axes[2, column].imshow(rgba, interpolation="nearest")
        axes[2, column].set_title(
            f"{name.capitalize()} · label {label} errors\nFP: {fp.sum()} px | FN: {fn.sum()} px", fontsize=10,
        )
    for axis in axes.flat:
        axis.axis("off")
    fig.legend(handles=[Patch(facecolor=c, label=f"Label {k}") for k, c in enumerate(colors)],
               loc="lower center", bbox_to_anchor=(.5, .038), ncol=6, frameon=False, fontsize=8)
    fig.legend(handles=[Patch(facecolor="#f22626", label=f"False positive · label {label}"),
                        Patch(facecolor="#1f7aff", label=f"False negative · label {label}")],
               loc="lower center", bbox_to_anchor=(.5, .008), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{case['filename']} | index {case['index']}\nLabel {label} FP/FN on resized 256 × 128 grid", fontsize=11)
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)
    return counts

def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_timing(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record["gpu_forward_argmax_ms"])
    return [{key: name, "slices": len(values), "gpu_seconds_sum": float(np.sum(values) / 1000),
             "mean_slice_ms": float(np.mean(values))} for name, values in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True,
                        help="Existing fixed-checkpoint metrics.json for the same split")
    parser.add_argument("--split", choices=["validate", "test"], default="test")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--indices", nargs="+", type=int, default=[268, 215, 271, 453, 515])
    parser.add_argument("--labels", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or any(label < 0 or label > 5 for label in args.labels):
        parser.error("Use warmup >= 1 and labels within 0–5")
    if len(set(args.indices)) != len(args.indices) or len(set(args.labels)) != len(args.labels):
        parser.error("Do not repeat indices or labels")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder is not empty; choose a new folder")
    reference = json.loads(args.reference_metrics.read_text())
    if reference["split"] != args.split or reference["partial_data"] or reference["smoke_checkpoint_used"]:
        parser.error("Reference must be a full evaluation of the same split using non-smoke checkpoints")
    import torch
    from dataset import HipMRIDataset
    from predict import confusion_matrix, defined_mean, json_values, load_model, metrics_from_confusion

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; submit inside a Slurm GPU job")
    dataset_args = {"split": args.split}
    if args.root is not None:
        dataset_args["root"] = args.root
    dataset = HipMRIDataset(**dataset_args)
    if len(dataset) != reference["evaluated_slices"]:
        raise ValueError("Dataset slice count differs from the reference evaluation")
    if any(index < 0 or index >= len(dataset) for index in args.indices):
        parser.error(f"Indices must be within 0–{len(dataset)-1}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overlays").mkdir()
    selected = {}
    slice_rows, scan_rows, case_rows, overlap_rows = [], [], [], []
    report = {
        "split": args.split, "evaluated_slices_per_model": len(dataset),
        "batch_size": 1, "precision": "float32", "warmup_calls_per_model": args.warmup,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "model_order": ["baseline", "improved"],
        "protocol": "One sequential pass per model; GPU forward + argmax timed with CUDA events and synchronization after warmup",
        "time_exclusions": ["NIfTI I/O", "preprocessing", "CPU/GPU transfers", "model loading", "plotting"],
        "aggregation": "Per-scan and per-case GPU seconds are sums of individual slice GPU times; not end-to-end application latency",
        "memory_protocol": "Peak allocated/reserved memory over batch-one inference, including validity checks; one model resident at a time",
        "case_overlap_note": "Supplementary case scores use batch-one float32; original test_best metrics are retained unchanged",
        "case_definition": "Case IDs and scan IDs parsed from filenames; scan ID includes week",
        "overlay_indices": args.indices, "overlay_labels": args.labels,
        "overlay_selection": "Illustrative error cases selected after evaluation; not an unbiased sample",
        "models": {}, "overlays": [],
    }
    device = torch.device("cuda")
    for name in ("baseline", "improved"):
        model, metadata = load_model(getattr(args, name), name, device)
        if metadata["sha256"] != reference["models"][name]["sha256"]:
            raise ValueError(f"{name} checkpoint differs from the fixed reference")
        print(f"Profiling {name}; selected checkpoint epoch {metadata['epoch']}", flush=True)
        model_records = []
        case_confusions = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
        with torch.inference_mode():
            warm_image, _ = dataset[0]
            warm_input = warm_image.unsqueeze(0).to(device)
            for _ in range(args.warmup):
                warm_logits = model(warm_input)
                warm_prediction = warm_logits.argmax(dim=1)
            torch.cuda.synchronize()
            del warm_input, warm_logits, warm_prediction
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for index in range(len(dataset)):
                image, target = dataset[index]
                filename = dataset.images[index].name
                case_id, scan_id, slice_number = scan_identifiers(filename)
                inputs = image.unsqueeze(0).to(device)
                torch.cuda.synchronize()
                start_event.record()
                logits = model(inputs)
                predicted = logits.argmax(dim=1)
                end_event.record()
                torch.cuda.synchronize()
                gpu_ms = float(start_event.elapsed_time(end_event))
                if tuple(logits.shape) != (1, 6, *target.shape) or not torch.isfinite(logits).all():
                    raise RuntimeError(f"Invalid logits: {name}, {filename}")
                prediction = predicted[0].cpu().numpy().astype(np.uint8)
                case_confusions[case_id] += confusion_matrix(target.numpy(), prediction)
                record = {"model": name, "index": index, "filename": filename,
                          "case_id": case_id, "scan_id": scan_id, "slice_number": slice_number,
                          "gpu_forward_argmax_ms": gpu_ms}
                model_records.append(record)
                if index in args.indices:
                    if index not in selected:
                        selected[index] = {"index": index, "filename": filename,
                                           "image": image[0].numpy().copy(), "target": target.numpy().copy()}
                    selected[index][name] = prediction.copy()
                del inputs, logits, predicted
                if (index + 1) % 100 == 0:
                    print(f"{name}: {index + 1}/{len(dataset)} slices", flush=True)
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        times = np.array([r["gpu_forward_argmax_ms"] for r in model_records])
        scans, cases = grouped_timing(model_records, "scan_id"), grouped_timing(model_records, "case_id")
        for row in scans + cases:
            row["model"] = name
        scan_rows.extend(scans); case_rows.extend(cases); slice_rows.extend(model_records)
        per_case_scores = []
        for case_id, confusion in sorted(case_confusions.items()):
            dice, iou = metrics_from_confusion(confusion)
            overlap_rows.append({"model": name, "case_id": case_id,
                "mean_dice_all_labels": defined_mean(dice), "mean_dice_labels_1_to_5": defined_mean(dice[1:]),
                **{f"dice_label_{k}": value for k, value in enumerate(json_values(dice))},
                **{f"iou_label_{k}": value for k, value in enumerate(json_values(iou))}})
            per_case_scores.append(defined_mean(dice))
        report["models"][name] = {
            **metadata, "mean_slice_ms": float(times.mean()), "median_slice_ms": float(np.median(times)),
            "p95_slice_ms": float(np.percentile(times, 95)), "total_gpu_seconds": float(times.sum()/1000),
            "mean_scan_gpu_seconds": float(np.mean([r["gpu_seconds_sum"] for r in scans])),
            "peak_allocated_gpu_gib": peak_allocated, "peak_reserved_gpu_gib": peak_reserved,
            "mean_of_case_mean_dice_all_labels": defined_mean(per_case_scores),
            "scans": scans, "cases": cases,
        }
        print(f"{name}: mean {times.mean():.3f} ms/slice; peak allocated {peak_allocated:.3f} GiB", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for index in args.indices:
        case = selected[index]
        for label in args.labels:
            relative = f"overlays/index_{index}_label_{label}.png"
            counts = save_overlay(case, label, args.output / relative)
            report["overlays"].append({"index": index, "filename": case["filename"], "label": label,
                                       "figure": relative, "pixel_errors": counts})
    write_csv(args.output / "per_slice_timing.csv", slice_rows,
              ["model", "index", "filename", "case_id", "scan_id", "slice_number", "gpu_forward_argmax_ms"])
    write_csv(args.output / "per_scan_timing.csv", scan_rows,
              ["model", "scan_id", "slices", "gpu_seconds_sum", "mean_slice_ms"])
    write_csv(args.output / "per_case_timing.csv", case_rows,
              ["model", "case_id", "slices", "gpu_seconds_sum", "mean_slice_ms"])
    write_csv(args.output / "per_case_overlap.csv", overlap_rows,
              ["model", "case_id", "mean_dice_all_labels", "mean_dice_labels_1_to_5"]
              + [f"dice_label_{k}" for k in range(6)] + [f"iou_label_{k}" for k in range(6)])
    (args.output / "profiling.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved profiling tables and {len(report['overlays'])} overlay figures to {args.output}", flush=True)


if __name__ == "__main__":
    main()