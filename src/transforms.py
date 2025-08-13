"""
Composable data-augmentation utilities for COnPose
We use TorchVision v0.18’s new functional transforms because they natively
support bounding boxes and keep gradients on tensors if needed.

"""

from __future__ import annotations
from typing import Dict, Tuple, Callable, List

import torch
import torchvision.transforms.v2 as T2
from torchvision.transforms.v2 import functional as F


class ColourJitter(T2.Transform):
    def __init__(
        self, brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.1
    ):
        super().__init__()
        self._t = T2.ColorJitter(brightness, contrast, saturation)

    def forward(self, img, target):
        return self._t(img), target


class RandomGaussianNoise(T2.Transform):
    def __init__(self, std: float = 0.02):
        super().__init__()
        self.std = std

    def forward(self, img, target):
        noise = torch.randn_like(img) * self.std
        img = (img + noise).clamp(0, 1)
        return img, target


class RandomHorizontalFlip(T2.Transform):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            h, w = img.shape[-2:]
            img = F.hflip(img)
            boxes = target["boxes"]
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
            target["boxes"] = boxes
            # no change to 3‑D pose
        return img, target


class RandomResizeCrop(T2.Transform):
    """
    Randomly zooms in (scale 0.9‑1.1) and pads back to original size,
    then clamps boxes to the cropped image.
    """

    def __init__(self, scale=(0.9, 1.1)):
        super().__init__()
        self.scale = scale

    def forward(self, img, target):
        h, w = img.shape[-2:]
        s = float(torch.empty((), device=img.device).uniform_(*self.scale))

        # resize + pad
        nh, nw = int(h * s), int(w * s)
        img = F.resize(img, [nh, nw], interpolation=T2.InterpolationMode.BILINEAR)
        pad_h = max(0, h - nh)
        pad_w = max(0, w - nw)
        img = F.pad(img, [0, 0, pad_w, pad_h])

        # crop back
        img = img[:, :h, :w]

        # scale and clamp boxes
        boxes = target["boxes"] * s
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

        # optional: drop degenerate boxes
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not keep.all():
            boxes = boxes[keep]
            # ↑ propagate to all parallel fields
            for k in (
                "labels",
                "poses",
                "dimensions",
                "canonical_keypoints",
                "class_names",
            ):
                if k in target and isinstance(target[k], torch.Tensor):
                    target[k] = target[k][keep]
                elif k in target and isinstance(target[k], list):
                    target[k] = [target[k][i] for i, m in enumerate(keep) if m]
        target["boxes"] = boxes
        return img, target


class RandomBoxOcclusion(T2.Transform):
    def __init__(self, p=0.3, occ_frac=(0.05, 0.2)):
        super().__init__()
        self.p = p
        self.occ_frac = occ_frac

    def forward(self, img, target):
        if torch.rand(1, device=img.device) >= self.p or len(target["boxes"]) == 0:
            return img, target

        # pick a random GT box
        idx = torch.randint(len(target["boxes"]), (), device=img.device).item()
        x1, y1, x2, y2 = target["boxes"][idx]
        bw = x2 - x1
        bh = y2 - y1

        # sample occlusion size as a fraction of box dims
        frac_w = (
            torch.rand((), device=img.device) * (self.occ_frac[1] - self.occ_frac[0])
            + self.occ_frac[0]
        )
        frac_h = (
            torch.rand((), device=img.device) * (self.occ_frac[1] - self.occ_frac[0])
            + self.occ_frac[0]
        )
        occ_w = bw * frac_w
        occ_h = bh * frac_h

        # sample occlusion centre within [x1, x2-occ_w] and [y1, y2-occ_h]
        cx = x1 + (x2 - occ_w - x1) * torch.rand((), device=img.device)
        cy = y1 + (y2 - occ_h - y1) * torch.rand((), device=img.device)

        # zero‑out the patch
        x0 = int(cx.item())
        y0 = int(cy.item())
        img[:, y0 : y0 + int(occ_h.item()), x0 : x0 + int(occ_w.item())] = 0.0

        return img, target


def training_transforms() -> Callable:
    """
    Returns a callable (img, target) -> (img, target) for the training split.
    """
    return T2.Compose(
        [
            T2.Resize((800, 800), interpolation=T2.InterpolationMode.BILINEAR),
            ColourJitter(0.1, 0.1, 0.1),
            RandomGaussianNoise(0.02),
            RandomBoxOcclusion(0.3, (0.05, 0.2)),
            # RandomHorizontalFlip(0.5),
            RandomResizeCrop((0.9, 1.1)),
            # Normalize to ImageNet mean/std for the RT‑DETR backbone
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def validation_transforms() -> Callable:
    """
    Minimal transform for val/test – just ImageNet normalisation.
    """
    return T2.Compose(
        [
            T2.Resize((800, 800), interpolation=T2.InterpolationMode.BILINEAR),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
