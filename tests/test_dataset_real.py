from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from src.dataset import ConPoseDataset, conpose_collate


@pytest.fixture(scope="session")
def real_conpose_dir():
    path = "/home/samuel/Downloads/small_set"
    if not path:
        pytest.skip(
            "Set CONPOSE_DATASET_DIR to your Unity-compact dataset root to run integration tests"
        )
    root = Path(path)
    images = root / "images"
    anns = root / "compact"
    if not images.exists() or not anns.exists():
        pytest.skip(f"{root} is not a valid ConPoseDataset directory")
    return root


def test_dataloader_and_collate(real_conpose_dir):
    ds = ConPoseDataset(real_conpose_dir)
    loader = DataLoader(ds, batch_size=2, collate_fn=conpose_collate)
    images, targets = next(iter(loader))

    # should get two entries
    assert isinstance(images, list) and len(images) == 2
    assert isinstance(targets, list) and len(targets) == 2

    # basic sanity
    for img in images:
        assert isinstance(img, torch.Tensor) and img.ndim == 3
    for t in targets:
        assert "boxes" in t and "labels" in t


def test_K_vs_Kraw(real_conpose_dir):
    ds = ConPoseDataset(real_conpose_dir)
    for img, t in (ds[i] for i in range(len(ds))):
        K_raw = t["K_raw"]
        cx_raw, cy_raw = K_raw[0, 2].item(), K_raw[1, 2].item()
        # look for a raw principal point at (0,0)
        if cx_raw == 0.0 and cy_raw == 0.0:
            K = t["K"]
            H, W = t["image_size"].tolist()
            # adjusted centre should be W/2, H/2
            assert K[0, 2] == pytest.approx(W / 2.0)
            assert K[1, 2] == pytest.approx(H / 2.0)
            break
    else:
        pytest.skip("No sample with raw (cx,cy)==(0,0) found")


def test_canonical_keypoints(real_conpose_dir):
    # pick first non‐empty sample to extract a class_id
    base_ds = ConPoseDataset(real_conpose_dir)
    _, base_t = next(
        (
            base_ds[i]
            for i in range(len(base_ds))
            if base_ds[i][1]["labels"].numel() > 0
        ),
        (None, None),
    )
    if base_t is None:
        pytest.skip("No objects in any sample")
    class_id = int(base_t["labels"][0].item())

    # supply a dummy kp mapping
    dummy_kp = torch.randn(5, 3)
    ds = ConPoseDataset(real_conpose_dir, canonical_keypoints={class_id: dummy_kp})
    _, t = ds[0]

    kps_list = t.get("canonical_keypoints")
    assert isinstance(kps_list, list)
    # find index of our class in the first sample
    idx = t["labels"].tolist().index(class_id)
    assert torch.equal(kps_list[idx], dummy_kp)


def test_real_dataset_load(real_conpose_dir):
    ds = ConPoseDataset(real_conpose_dir, assume_normalised_intrinsics=True)
    # must have at least one sample
    assert len(ds) > 0, "No samples found in real dataset"

    # randomly sample a few items
    for idx in range(min(len(ds), 5)):
        img, target = ds[idx]

        # image tensor
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3 and img.shape[0] == 3

        # boxes & labels
        boxes = target["boxes"]
        labels = target["labels"]
        assert boxes.ndim == 2 and boxes.shape[1] == 4
        assert boxes.shape[0] == labels.numel()

        # non-degenerate boxes
        x1, y1, x2, y2 = boxes.unbind(1)
        assert torch.all(x2 > x1) and torch.all(y2 > y1)

        # poses & dimensions
        poses = target["poses"]
        dims = target["dimensions"]
        assert poses.ndim == 2 and poses.shape[1] == 7
        assert dims.ndim == 2 and dims.shape[1] == 3

        # quaternion normalisation
        qw, qx, qy, qz, *_ = poses[0]
        norm = qw**2 + qx**2 + qy**2 + qz**2
        assert pytest.approx(1.0, rel=1e-5) == norm.item()

        # intrinsics sanity
        K = target["K"]
        H, W = target["image_size"].tolist()
        # principal point near centre
        assert 0.4 * W < K[0, 2] < 0.6 * W
        assert 0.4 * H < K[1, 2] < 0.6 * H


def test_every_json_has_image(real_conpose_dir):
    # ensure no orphan JSONs or images
    images = {p.name for p in (real_conpose_dir / "images").glob("*.png")}
    jsons = {p.stem + ".png" for p in (real_conpose_dir / "compact").glob("*.json")}
    missing = jsons - images
    assert not missing, f"Missing images for annotations: {missing}"
