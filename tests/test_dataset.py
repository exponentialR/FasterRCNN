import json
import pytest
import torch
from PIL import Image

from src.dataset import ConPoseDataset, conpose_collate


@pytest.fixture
def tmp_conpose_dir(tmp_path):
    root = tmp_path / "ConPoseSample"
    imgs = root / "images"
    anns = root / "compact"
    imgs.mkdir(parents=True)
    anns.mkdir(parents=True)

    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    img_file = imgs / "test_run_rgb_frame_000000.png"
    img.save(img_file)

    # Dump a matching annotation JSON
    ann = {
        "camera_intrinsics": {
            "fx": 0.5,
            "fy": 0.5,
            "cx": 0.0,
            "cy": 0.0,
            "width": 640,
            "height": 480,
        },
        "image_filename": "test_run_rgb_frame_000000.png",
        "objects": [
            {
                "class_id": 2,
                "class_name": "TestObject",
                "2D_center": [0.5, 0.5],
                "width": 0.25,
                "height": 0.25,
                "6DOF_pose": {
                    "position": [1.0, 2.0, 3.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "dimensions": [0.1, 0.2, 0.3],
                },
            }
        ],
    }
    ann_file = anns / "test_run_rgb_frame_000000.json"
    ann_file.write_text(json.dumps(ann, indent=2))

    return root


def test_dataset_single(tmp_conpose_dir):
    ds = ConPoseDataset(tmp_conpose_dir, assume_normalised_intrinsics=True)
    assert len(ds) == 1

    img, target = ds[0]
    # Image tensor should be (3, H, W)
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 480, 640)
    assert img.max() <= 1.0 and img.min() >= 0.0

    # Check boxes
    boxes = target["boxes"]
    assert isinstance(boxes, torch.Tensor)
    assert boxes.shape == (1, 4)
    # Should be centred at (320,240) with w=160,h=120
    x1, y1, x2, y2 = boxes[0].tolist()
    assert pytest.approx(320 - 80, abs=1e-3) == x1
    assert pytest.approx(240 - 60, abs=1e-3) == y1
    assert pytest.approx(320 + 80, abs=1e-3) == x2
    assert pytest.approx(240 + 60, abs=1e-3) == y2

    # Check labels
    labels = target["labels"]
    assert torch.equal(labels, torch.tensor([2], dtype=torch.int64))

    # Check poses and dims
    poses = target["poses"]
    dims = target["dimensions"]
    assert poses.shape == (1, 7)
    assert dims.shape == (1, 3)
    # Quaternion should be normalised
    qw, qx, qy, qz, tx, ty, tz = poses[0].tolist()
    assert pytest.approx(qw**2 + qx**2 + qy**2 + qz**2, rel=1e-5) == 1.0
    assert (tx, ty, tz) == (1.0, 2.0, 3.0)

    # Check intrinsics
    K = target["K"]
    assert K.shape == (3, 3)
    # fx_px = 0.5*640=320, fy_px=0.5*480=240, cx=320, cy=240
    assert K[0, 0] == pytest.approx(320)
    assert K[1, 1] == pytest.approx(240)
    assert K[0, 2] == pytest.approx(320)
    assert K[1, 2] == pytest.approx(240)


def test_collate():
    # Simple collate on one dummy sample
    dummy_img = torch.zeros((3, 10, 10))
    dummy_target = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }
    images, targets = conpose_collate([(dummy_img, dummy_target)])
    assert isinstance(images, list) and isinstance(targets, list)
    assert images[0].shape == (3, 10, 10)
    assert "boxes" in targets[0]
