from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Literal
import json

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from src.transforms import training_transforms, validation_transforms
from scripts.utils import load_canonical_keypoints, compute_dataset_translation_norm


def _fmt_pts(X: torch.Tensor, ndigits: int = 6) -> list[list[float]]:
    """Round and return as nested lists for readable printing."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu()
    return [[round(float(v), ndigits) for v in row] for row in X.tolist()]


def _quat_from_R(R: torch.Tensor) -> torch.Tensor:
    """SO(3)->quat (w,x,y,z). R: (3,3)."""
    tr = R.trace()
    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return torch.stack([qw, qx, qy, qz])


def _R_stats(R: torch.Tensor) -> tuple[float, float]:
    """Return (det(R), ||R^T R - I||_F)."""
    det = torch.det(R).item()
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    ortho = (R.transpose(-1, -2) @ R - I).norm().item()
    return det, ortho


def _load_image(path: Path) -> Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _xyzw_to_wxyz(q_xyzw: Tensor) -> Tensor:
    """Unity quaternions are [x,y,z,w]; convert to (w,x,y,z) and normalise."""
    x, y, z, w = q_xyzw.unbind(-1)
    q = torch.stack([w, x, y, z], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    return q


def _quat_wxyz_to_R(q_wxyz: Tensor) -> Tensor:
    """Quaternion (w,x,y,z) -> rotation matrix, batched or single."""
    w, x, y, z = q_wxyz.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(
        q_wxyz.shape[:-1] + (3, 3)
    )
    return R


def _project_point_cam(K: Tensor, Xc: Tensor) -> Tensor:
    """Project a single 3D point in camera frame to pixels. K: (3,3), Xc: (...,3)."""
    X, Y, Z = Xc[..., 0], Xc[..., 1], Xc[..., 2].clamp(min=1e-6)
    u = K[0, 0] * X / Z + K[0, 2]
    v = K[1, 1] * Y / Z + K[1, 2]
    return torch.stack([u, v], dim=-1)


def _load_canonical_keypoints(directory: Path) -> Dict[str, Tensor]:
    mapping: Dict[str, Tensor] = {}
    for jf in sorted(directory.glob("*.json")):
        jd = json.loads(jf.read_text())
        kpts = jd.get("keypoints", {})
        items = sorted(kpts.items(), key=lambda x: x[0])
        coords = torch.tensor(
            [[v["x"], v["y"], v["z"]] for _, v in items], dtype=torch.float32
        )
        mapping[jf.stem.lower()] = coords
    return mapping


# ------------------------------- dataset -------------------------------


class ConPoseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        canonical_keypoints: Optional[Dict[str, Tensor]] = None,
        assume_normalised_intrinsics: bool = True,
        annotation_dir_name: str = "compact",
        use_canonical_dir: bool = False,
        gt_frame_mode: Literal["auto", "camera", "world"] = "camera",
        normalise_translation: bool = False,
        debug: bool = False,
        debug_max_items: int = 5,
        debug_print_canon: bool = False,
    ) -> None:
        """
        root: dataset split dir containing images/ and <annotation_dir_name>/.
        canonical_keypoints: per-class canonical (object-local) keypoints.
        assume_normalised_intrinsics: if True, (fx,fy,cx,cy) are fractions of (W,H).
        gt_frame_mode:
            - "auto"   : decide per instance whether pose is camera- or world-frame (by reprojection)
            - "camera" : trust pose['position','rotation'] is already object->camera (Unity sensor frame)
            - "world"  : treat pose as world frame and convert to object->camera using camera_extrinsics
        normalise_translation: if True, z-score t using dataset stats (off by default).
        """
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / annotation_dir_name
        self.transform = transform
        self.assume_normalised_intrinsics = assume_normalised_intrinsics
        self.gt_frame_mode = gt_frame_mode
        self.normalise_translation = normalise_translation
        self.debug = debug
        self.debug_max_items = debug_max_items
        self._debug_emitted = 0
        self.debug_print_canon = debug_print_canon

        # Canonicals
        if use_canonical_dir:
            candir = self.root.parent / "canonical_files"
            if not candir.exists():
                raise RuntimeError(
                    f"Canonical keypoints directory {candir} does not exist."
                )
            self.canonical_keypoints = _load_canonical_keypoints(candir)
        else:
            self.canonical_keypoints = canonical_keypoints or {}

        if self.normalise_translation:
            self.translation_mean, self.translation_std = (
                compute_dataset_translation_norm(self.ann_dir)
            )
        else:
            self.translation_mean = self.translation_std = None

        json_files = sorted(self.ann_dir.glob("*.json"))
        if split is not None:
            stems = set(split)
            json_files = [p for p in json_files if p.stem in stems]
        if not json_files:
            raise RuntimeError(f"No annotation JSON found in {self.ann_dir}")
        self.samples = json_files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        jf = self.samples[index]
        meta = json.loads(jf.read_text())

        # --- image ---
        image_path = self.images_dir / meta["image_filename"]
        image = _load_image(image_path)
        H = int(meta["camera_intrinsics"]["height"])
        W = int(meta["camera_intrinsics"]["width"])

        # --- intrinsics ---
        intr = meta["camera_intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        if self.assume_normalised_intrinsics:
            fx_px, fy_px = fx * W, fy * H
            cx_px, cy_px = cx * W, cy * H
        else:
            fx_px, fy_px, cx_px, cy_px = fx, fy, cx, cy

        if cx_px == 0 and cy_px == 0:
            cx_px, cy_px = W / 2.0, H / 2.0

        K = torch.tensor(
            [[fx_px, 0.0, cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        # --- camera extrinsics (world) ---
        cam_ex = meta.get("camera_extrinsics", None)
        if cam_ex is not None:
            t_wc = torch.tensor(cam_ex["position"], dtype=torch.float32)  # (3,)
            q_wc_xyzw = torch.tensor(
                cam_ex["rotation"], dtype=torch.float32
            )  # [x,y,z,w]
            q_wc_wxyz = _xyzw_to_wxyz(q_wc_xyzw)
            R_wc = _quat_wxyz_to_R(q_wc_wxyz)  # (3,3)
        else:
            t_wc = None
            R_wc = None

        # --- iterate objects ---
        boxes: List[List[float]] = []
        labels: List[int] = []
        poses: List[List[float]] = []  # (qw,qx,qy,qz, tx,ty,tz) in CAMERA frame
        dims: List[List[float]] = []
        class_names: List[str] = []
        canonical_keypoints: List[Tensor] = []

        for obj in meta.get("objects", []):
            class_id = int(obj["class_id"])
            class_name = obj.get("class_name", str(class_id))
            cname_key = class_name.lower()

            # 2D box from centre+size (centre given normalised in JSON)
            cx2d = float(obj["2D_center"][0]) * W
            cy2d = float(obj["2D_center"][1]) * H
            bw = float(obj["width"]) * W
            bh = float(obj["height"]) * H
            x1, y1 = max(0.0, cx2d - bw / 2.0), max(0.0, cy2d - bh / 2.0)
            x2, y2 = min(W, cx2d + bw / 2.0), min(H, cy2d + bh / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue

            # Raw pose from JSON (Unity order for quats)
            pose = obj.get("6DOF_pose", None)
            if pose is None:
                continue

            t_raw = torch.tensor(
                pose.get("position", [0.0, 0.0, 0.0]), dtype=torch.float32
            )
            q_raw_xyzw = torch.tensor(
                pose.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=torch.float32
            )
            q_raw_wxyz = _xyzw_to_wxyz(q_raw_xyzw)
            R_raw = _quat_wxyz_to_R(q_raw_wxyz)

            # Candidate A: assume pose is already camera-frame (object->camera)
            R_cam = R_raw
            t_cam = t_raw

            # Candidate B: assume pose is world-frame; convert using camera extrinsics
            use_world = (self.gt_frame_mode == "world") or (
                self.gt_frame_mode == "auto" and cam_ex is not None
            )
            if use_world and (R_wc is not None):
                R_wo = R_raw
                t_wo = t_raw
                R_co_from_world = R_wc.T @ R_wo
                t_co_from_world = R_wc.T @ (t_wo - t_wc)
                R_world, t_world = R_co_from_world, t_co_from_world
            else:
                R_world, t_world = None, None

            if self.gt_frame_mode == "camera" or R_world is None:
                R_final, t_final, src = R_cam, t_cam, "camera"
            elif self.gt_frame_mode == "world":
                R_final, t_final, src = R_world, t_world, "world"
            else:  # auto
                uv_cam = _project_point_cam(K, t_cam)
                uv_world = _project_point_cam(K, t_world)
                uv_gt = torch.tensor([cx2d, cy2d], dtype=torch.float32)
                e_cam = (uv_cam - uv_gt).pow(2).sum().sqrt()
                e_world = (uv_world - uv_gt).pow(2).sum().sqrt()

                z_cam, z_world = t_cam[2], t_world[2]
                if z_cam <= 0:
                    e_cam = e_cam + 1e6
                if z_world <= 0:
                    e_world = e_world + 1e6
                if e_world + 1e-6 < e_cam:
                    R_final, t_final, src = R_world, t_world, "world"
                else:
                    R_final, t_final, src = R_cam, t_cam, "camera"
            X_can = self.canonical_keypoints.get(
                cname_key, torch.zeros((0, 3), dtype=torch.float32)
            )
            if self.debug and self._debug_emitted < self.debug_max_items:
                if X_can.numel():
                    mu = X_can.mean(0)  # (3,)
                    uv_cam = _project_point_cam(K, R_cam @ mu + t_cam)
                    uv_world = (
                        _project_point_cam(K, R_world @ mu + t_world)
                        if (R_world is not None)
                        else torch.tensor([float("inf"), float("inf")])
                    )
                else:
                    mu = torch.zeros(3)
                    uv_cam = _project_point_cam(K, t_cam)
                    uv_world = (
                        _project_point_cam(K, t_world)
                        if (t_world is not None)
                        else torch.tensor([float("inf"), float("inf")])
                    )

                Xc = self.canonical_keypoints.get(cname_key, None)
                if Xc is not None and Xc.numel():
                    mu = Xc.mean(0)  # (3,)
                    uv_cam = _project_point_cam(K, R_cam @ mu + t_cam)
                    uv_world = (
                        _project_point_cam(K, R_world @ mu + t_world)
                        if (R_world is not None)
                        else torch.tensor([float("inf"), float("inf")])
                    )
                else:
                    uv_cam = _project_point_cam(K, t_cam)
                    uv_world = (
                        _project_point_cam(K, t_world)
                        if (t_world is not None)
                        else torch.tensor([float("inf"), float("inf")])
                    )

                uv_gt = torch.tensor([cx2d, cy2d], dtype=torch.float32)
                e_cam = (uv_cam - uv_gt).norm().item()
                e_world = (uv_world - uv_gt).norm().item()

                detR, ortho = _R_stats(R_final)
                q_dbg = _quat_from_R(R_final)  # (w,x,y,z)

                # quick stats for the canonical
                c_shape = tuple(X_can.shape)
                c_mean = X_can.mean(0).tolist() if X_can.numel() else [0.0, 0.0, 0.0]

                print(
                    f"[{jf.name}] class_id={class_id} class='{class_name}' src={src}  "
                    f"Z={t_final[2].item():.3f}  det(R)={detR:.4f}  ||R^TR-I||={ortho:.2e}  "
                    f"e_cam={e_cam:.1f}  e_world={e_world:.1f}  "
                    f"q(wxyz)={[round(v,4) for v in q_dbg.tolist()]}  "
                    f"t={ [round(v,4) for v in t_final.tolist()] }  "
                    f"canon_shape={c_shape}  canon_mean={[round(v,4) for v in c_mean]}"
                )

                if self.debug_print_canon:
                    print(
                        f"[CANON] class_id={class_id}  class='{class_name}'  K={X_can.shape[0]}"
                    )
                    print(_fmt_pts(X_can, ndigits=6))

                self._debug_emitted += 1

            # R = R_final
            # tr = R.trace()
            q_cam_wxyz = _quat_from_R(R_final)  # (w,x,y,z)

            t_cam_wxyz = t_final  # already camera frame

            if self.normalise_translation:
                t_cam_wxyz = (t_cam_wxyz - self.translation_mean) / (
                    self.translation_std + 1e-8
                )

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id - 1)  # 1..C -> 0..C-1
            class_names.append(class_name)
            poses.append(torch.cat([q_cam_wxyz, t_cam_wxyz], dim=0).tolist())
            dims_xyz = pose.get("dimensions", [0.0, 0.0, 0.0])
            dims.append([float(d) for d in dims_xyz])

            if cname_key in self.canonical_keypoints:
                canonical_keypoints.append(self.canonical_keypoints[cname_key])
            else:
                canonical_keypoints.append(torch.zeros((0, 3), dtype=torch.float32))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "poses": torch.tensor(
                poses, dtype=torch.float32
            ),  # (qw,qx,qy,qz, tx,ty,tz) in CAMERA frame
            "dimensions": torch.tensor(dims, dtype=torch.float32),
            "class_names": class_names,
            "canonical_keypoints": canonical_keypoints,
            "K": K,
            "image_size": torch.tensor([H, W], dtype=torch.int64),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


def conpose_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ----------------------------------------------------------------------
# Quick local test:  python -m src.dataset --root /path/to/dataset
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument(
        "--gt_frame_mode", choices=["auto", "camera", "world"], default="auto"
    )
    ap.add_argument("--assume_norm_intrinsics", action="store_true", default=True)
    ap.add_argument("--audit", type=int, default=20, help="number of objects to print")
    args = ap.parse_args()

    canonical_dir = Path(args.data_root) / "canonical_files/"
    print(f' DATA_ROOT = "{args.data_root}"')
    print(f"Loading canonical keypoints from {canonical_dir}")
    val_dir = Path(args.data_root)  # / "val"

    canonical_map = load_canonical_keypoints(canonical_dir)

    ds_train = ConPoseDataset(
        root=val_dir,
        canonical_keypoints=canonical_map,
        assume_normalised_intrinsics=args.assume_norm_intrinsics,
        gt_frame_mode="camera",
        debug=True,
        debug_max_items=args.audit,
        transform=None,
    )

    printed = 0
    for i in range(len(ds_train)):
        _, tgt = ds_train[i]
        printed = ds_train._debug_emitted
        if printed >= args.audit:
            break

    all_t = []
    for i in range(min(len(ds_train), 200)):  # sample first 200 images for speed
        _, tgt = ds_train[i]
        if tgt["poses"].numel():
            all_t.append(tgt["poses"][:, 4:7])  # (N,3)
    if all_t:
        T = torch.cat(all_t, dim=0)
        mu, sd = T.mean(0), T.std(0)
        print(f"\nTranslation stats (camera frame) over {T.shape[0]} instances:")
        print(f"  mean = {mu.tolist()}")
        print(f"  std  = {sd.tolist()}")

    # for name, ds in [("train", ds_train), ("val", ds_val)]:
    #     img, tgt = ds[0]
    #     H, W = img.shape[-2:]
    #     print(
    #         f"{name}: img {tuple(img.shape)}, "
    #         f"{len(tgt['boxes'])} boxes, "
    #         f"first box {tgt['boxes'][0].tolist() if tgt['boxes'].numel() else 'n/a'}"
    #     )
    #     # Get the
    #     # sanity: boxes inside image
    #     assert (tgt["boxes"][:, 0] >= 0).all() and (tgt["boxes"][:, 2] <= W).all()
    #     assert (tgt["boxes"][:, 1] >= 0).all() and (tgt["boxes"][:, 3] <= H).all()
    # print("✓ Dataset + transforms look good.")
