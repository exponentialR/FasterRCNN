"""
Quick sanity‑check: run detector → PoseInit → PoseRef → `evaluate`
on a single validation batch and print the metrics.
"""

from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import ConPoseDataset, conpose_collate
from src.detection import RTDETRDetector
from src.pose_initialisation import PoseInit
from src.refinement import PoseRefinement
from src.metrics import evaluate
from src.metrics import ModelDatabase  # just to assert meshes load OK
from .utils import load_canonical_keypoints


# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--kp_dir", required=True, type=Path)
    p.add_argument("--mesh_dir", required=True, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # ‑‑ canonical templates & dataset
    canonical_map = load_canonical_keypoints(args.kp_dir)
    ds_val = ConPoseDataset(root=args.data_root, canonical_keypoints=canonical_map)
    val_loader = DataLoader(
        ds_val, batch_size=2, shuffle=False, num_workers=2, collate_fn=conpose_collate
    )

    # ‑‑ models
    detector = RTDETRDetector(num_classes=6).to(device).eval()
    pose_init = PoseInit(in_dim=512).to(device).eval()
    pose_ref = PoseRefinement(feat_dim=512).to(device).eval()

    # take first mini‑batch only
    subset_loader = [next(iter(val_loader))]
    metrics = evaluate(
        detector,
        pose_init,
        pose_ref,
        subset_loader,
        canonical_map=canonical_map,
        t_mean=ds_val.translation_mean,
        t_std=ds_val.translation_std,
        mesh_dir=args.mesh_dir,
    )
    print("\nDemo evaluation on one batch:")
    for k, v in metrics.items():
        if "add" in k or "acc" in k or "map" in k:
            print(f"  {k:16s} : {v*100:6.2f} %")
        elif "rot" in k:
            print(f"  {k:16s} : {v:6.2f} °")
        else:
            print(f"  {k:16s} : {v:6.4f}")

    # simple assert meshes load
    _ = ModelDatabase(args.mesh_dir)


if __name__ == "__main__":
    main()
