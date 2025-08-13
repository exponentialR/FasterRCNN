"""
Training: Faster-RPN Detector standalone on ConPose data
"""

import argparse, time
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch import nn, optim

from src.dataset import ConPoseDataset, conpose_collate
from src.faster_rpn_detector import ObjectModule
from scripts.utils import count_parameters


def to_torchvision_format(batch_targets, device, add_background: bool = False):
    """
    Convert ConPose targets to Torchvision format.
    If add_background=True, shift labels by +1 (torchvision training convention).
    """
    tv = []
    for t in batch_targets:
        tv.append(
            {
                "boxes": t["boxes"].to(device),  # (N,4)
                "labels": t["labels"].to(device) + (1 if add_background else 0),  # (N,)
            }
        )
    return tv


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device) -> tuple[float, float]:
    metric = MeanAveragePrecision(iou_thresholds=[0.50, 0.75]).to(device)
    model.eval()

    old_thresh = float(model.model.roi_heads.score_thresh)
    model.model.roi_heads.score_thresh = 0.05

    for imgs, tgts in val_loader:
        imgs = [img.to(device) for img in imgs]
        preds = model(imgs)  # labels are 0..C-1 by our ObjectModule

        metric.update(
            [
                {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
                for p in preds
            ],
            to_torchvision_format(
                tgts, device, add_background=False
            ),  # << no +1 at eval
        )

    # restore
    model.model.roi_heads.score_thresh = old_thresh

    m = metric.compute()
    return m["map_50"].item(), m["map_75"].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--bs", default=8, type=int)
    p.add_argument(
        "--lr", default=2e-3, type=float
    )  # a bit higher for SGD; step it down if unstable
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--num_classes", default=7, type=int)
    args = p.parse_args()

    dev = torch.device(args.device)
    data_root = Path(args.data_root)
    train_ds = ConPoseDataset(root=data_root / "train")
    val_ds = ConPoseDataset(root=data_root / "val")

    train_ld = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=conpose_collate,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=conpose_collate,
        pin_memory=True,
    )

    # Detector
    model = ObjectModule(num_classes=args.num_classes).to(dev)
    model_details = [("Faster RPN", model)]
    rows, sum_total, sum_train = [], 0, 0
    for name, mdl in model_details:
        total_params, trainable_params = count_parameters(mdl)
        rows.append((name, total_params, trainable_params))
        sum_total += total_params
        sum_train += trainable_params
    rows.append(("Total", sum_total, sum_train))

    print(f"{'Model':<14} {'Total (M)':>10} {'Trainable (M)':>15}")
    print("-" * 42)
    for name, total, trainable in rows:
        print(f"{name:<14} {total / 1e6:10.2f} {trainable / 1e6:15.2f}")
    print("-" * 42)

    optimiser = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    best_map = 0.0
    ckpt_dir = Path("runs/only_detector")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, tgts in train_ld:
            imgs = [img.to(dev) for img in imgs]
            tv_tgts = to_torchvision_format(
                tgts, dev, add_background=False
            )  # << +1 for training

            loss_dict = model(imgs, tv_tgts)
            loss = sum(loss_dict.values())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running_loss += loss.item()

        val_map_50, val_map_75 = evaluate(model, val_ld, dev)
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"Epoch {epoch:02d} | loss {running_loss/len(train_ld):.4f} "
            f"| mAP@0.50 {val_map_50*100:5.2f}% "
            f"| mAP@0.75 {val_map_75*100:5.2f}% "
        )

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # save “best so far” w.r.t. mAP@0.50
        if val_map_50 > best_map:
            best_map = val_map_50
            torch.save(
                model.state_dict(),
                ckpt_dir / f"faster_rpn_best_{time_str}_epoch{epoch:02d}.pth",
            )

        # save every epoch as “last”
    torch.save(model.state_dict(), ckpt_dir / "faster_rpn_last.pth")


if __name__ == "__main__":
    main()
