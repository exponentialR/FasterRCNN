#!/usr/bin/env python
"""
Evaluate a saved Faster-RPN checkpoint on a ConPose validation set.

Usage
-----
python -m scripts.eval_faster_rpn \
    --val_root   /data/conpose/val \
    --ckpt_path  runs/faster_rpn/best_detector.pth \
    --batch_size 4
"""

from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.dataset import ConPoseDataset, conpose_collate
from src.faster_rpn_detector import ObjectModule


# ─── helper: shift labels +1 for TorchVision / TorchMetrics ─────────────
def to_tv_targets(batch_targets, device):
    tv = []
    for t in batch_targets:
        tv.append(
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device) + 1,  # 1 … C, 0 is BG
            }
        )
    return tv


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--val_root",
        required=True,
        type=Path,
        help="Validation split root (images/ & compact/)",
    )
    p.add_argument(
        "--ckpt_path",
        required=True,
        type=Path,
        help="Path to *.pth with model.state_dict()",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # dataset / loader ----------------------------------------------------
    val_ds = ConPoseDataset(root=args.val_root)
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=conpose_collate,
    )

    # model ---------------------------------------------------------------
    detector = ObjectModule(num_classes=7).to(device)  # 6 steel + BG
    state = torch.load(args.ckpt_path, map_location="cpu")
    detector.load_state_dict(state)
    detector.eval()

    metric = MeanAveragePrecision(  # COCO style 0.50:0.95 + per-class
        iou_thresholds=None,  # defaults to 0.50:0.95 step 0.05
        class_metrics=True,
    ).to(device)

    # loop ----------------------------------------------------------------
    for imgs, tgts in val_ld:
        imgs = [im.to(device) for im in imgs]
        preds = detector(imgs)  # list[dict]
        metric.update(
            [
                {
                    "boxes": p["boxes"],
                    "scores": p["scores"],
                    "labels": p["labels"],
                }
                for p in preds
            ],
            to_tv_targets(tgts, device),
        )

    res = metric.compute()
    print("\n────────  Detection metrics  ────────")
    print(f"mAP  @[0.50:0.95] : {res['map']    *100:6.2f}%")
    print(f"mAP  @0.50        : {res['map_50'] *100:6.2f}%")
    print(f"mAP  @0.75        : {res['map_75'] *100:6.2f}%")
    print("\nPer-class AP (0.50:0.95)")
    for cls_id, ap in enumerate(res["map_per_class"].tolist(), start=1):
        print(f"  class {cls_id:2d}: {ap*100:6.2f}%")


if __name__ == "__main__":
    main()
