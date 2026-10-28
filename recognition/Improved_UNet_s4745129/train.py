"""Training loop adapted from my Demo 2 segmentation work."""

import argparse
import json
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, NUM_CLASSES, masks_to_one_hot
from modules import UNET, ImprovedUNet


def dice_loss(logits, targets):
    # Calculate Dice in float32 for numerical stability
    probs = torch.softmax(logits.float(), dim=1)
    one_hot = masks_to_one_hot(targets)

    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))

    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice.mean()


def train_epoch(loader, model, optimizer, scaler, device, max_batches=None):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    consecutive_overflows = 0
    batches = loader if max_batches is None else islice(loader, max_batches)

    for step, (images, masks) in enumerate(batches, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Forward pass using mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = ce_loss_fn(logits, masks) + dice_loss(logits, masks)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        # Backward pass and check gradients before updating weights
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Check all gradients before attempting to clip them
        gradients_finite = torch.stack([
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        ]).all()

        if gradients_finite:
            consecutive_overflows = 0

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        else:
            consecutive_overflows += 1

            print(
                f"Step {step}: non-finite gradients; "
                f"AMP will skip this update. Scale={scaler.get_scale()}",
                flush=True,
            )

            # Stop if this becomes persistent rather than occasional
            if consecutive_overflows >= 10:
                raise RuntimeError("10 consecutive non-finite gradient batches")

        # unscale_ already recorded any non-finite gradients.
        # GradScaler skips unsafe updates and adjusts its scale.
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if step == 1 or step % 25 == 0 or max_batches is not None:
            print(f"Step {step}: loss={loss.item():.4f}", flush=True)

    return total_loss / total_samples


@torch.no_grad()
def validate(loader, model, device, max_batches=None):
    model.eval()

    # Rows are true labels, columns are predicted labels
    confusion = torch.zeros(
        NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device
    )
    total_loss = 0.0
    total_samples = 0
    ce_loss_fn = nn.CrossEntropyLoss()
    batches = loader if max_batches is None else islice(loader, max_batches)

    for images, masks in batches:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = ce_loss_fn(logits, masks) + dice_loss(logits, masks)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")

        predictions = logits.argmax(dim=1)
        counts = torch.bincount(
            (NUM_CLASSES * masks + predictions).reshape(-1),
            minlength=NUM_CLASSES ** 2,
        )
        confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    confusion = confusion.double()
    intersection = confusion.diag()
    true_count = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    denominator = true_count + predicted_count
    union = denominator - intersection

    # Leave completely absent classes undefined rather than scoring them as 1
    dice = torch.where(
        denominator > 0,
        2 * intersection / denominator,
        torch.nan,
    )
    iou = torch.where(union > 0, intersection / union, torch.nan)

    return total_loss / total_samples, dice.cpu(), iou.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
    "--model",
    choices=["baseline", "improved"],
    default="baseline",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid epochs, batch size or worker count")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Run inside a Slurm GPU job.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")

    # Separate smoke-test files from full training results
    # Separate results by model and run type
    output = args.output if args.output is not None else (
        Path("results") / args.model
    )

    if args.smoke:
        output = output / "smoke"

    output.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        HipMRIDataset(split="train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        HipMRIDataset(split="validate"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Select architecture; keep training settings the same
    if args.model == "baseline":
        model = UNET(in_channels=1, out_channels=NUM_CLASSES)
    else:
        model = ImprovedUNet(in_channels=1, out_channels=NUM_CLASSES)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    # Keep a copy to verify that training changes model parameters
    first_parameter = next(model.parameters())
    initial_weights = first_parameter.detach().clone()

    epochs = 1 if args.smoke else args.epochs
    history = []
    best_dice = float("-inf")
    checkpoint_path = output / "best_unet.pt"

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Smoke test:", args.smoke, flush=True)
    print("Model:", args.model, flush=True)
    print("Parameters:", sum(p.numel() for p in model.parameters()), flush=True)

    for epoch in range(1, epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()

        train_loss = train_epoch(
            train_loader, model, optimizer, scaler, device,
            max_batches=5 if args.smoke else None,
        )
        val_loss, dice, iou = validate(
            val_loader, model, device,
            max_batches=2 if args.smoke else None,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
        mean_dice = torch.nanmean(dice).item()

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train loss: {train_loss:.4f}; val loss: {val_loss:.4f}")
        for label in range(NUM_CLASSES):
            print(
                f"Label {label}: Dice={dice[label]:.4f}, "
                f"IoU={iou[label]:.4f}"
            )
        print(f"Time: {elapsed:.1f}s; peak allocated GPU memory: {peak_gib:.2f} GiB")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice": [v if torch.isfinite(torch.tensor(v)) else None
                     for v in dice.tolist()],
            "iou": [v if torch.isfinite(torch.tensor(v)) else None
                    for v in iou.tolist()],
            "seconds": elapsed,
            "peak_allocated_gpu_gib": peak_gib,
        })

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "mean_dice": mean_dice,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "seed": 42,
                    "smoke": args.smoke,
                    "model": args.model,
                },
            }, checkpoint_path)
            print("Saved checkpoint:", checkpoint_path, flush=True)

        (output / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False)
        )

    if torch.equal(initial_weights, first_parameter.detach()):
        raise RuntimeError("Checked model parameters did not change")

    print("Parameter update check passed.")

    # Verify saved checkpoint reproduces predictions after loading
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    example, _ = next(iter(val_loader))
    example = example[:1].to(device)

    with torch.no_grad():
        expected = model(example).clone()
        first_parameter.add_(1)
        model.load_state_dict(checkpoint["state_dict"])
        restored = model(example)

    torch.testing.assert_close(restored, expected)
    print("Checkpoint reload check passed.")

    if args.smoke:
        print("Smoke test passed. Metrics are from partial data, not final results.")


if __name__ == "__main__":
    main()