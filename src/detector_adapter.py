from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from torch import Tensor
from torchvision.ops import MultiScaleRoIAlign


@torch.no_grad()
def _rescale_boxes_to_resized(
    boxes_xyxy: Tensor, original_size: Tuple[int, int], resized_size: Tuple[int, int]
) -> Tensor:
    """
    Rescale boxes from original image size to resized image size.

    Args:
        boxes_xyxy: Tensor of shape (N, 4) with boxes in (x1, y1, x2, y2) format.
        original_size: Tuple of (height, width) of the original image.
        resized_size: Tuple of (height, width) of the resized image.

    Returns:
        Rescaled boxes in (x1, y1, x2, y2) format.
    """
    orig_h, orig_w = original_size
    resized_h, resized_w = resized_size
    scale_x = resized_w / float(orig_w)
    scale_y = resized_h / float(orig_h)
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y
    return boxes_xyxy


class DetectorAdapter(torch.nn.Module):
    """
    Wraps a Faster R-CNN model:
    - runs detection on images (eval mode)
    - shifts labels from 1...C -> 0...C-1
    - pools ROI features for the kept detections
        using the model's own transform, roi_pool, and box_head -> then projects to 512-d
    Output per image:
    {'boxes' : Tensor (N, 4),  # xyxy
        'scores': Tensor (N,),  # 0...1
     'labels': Tensor (N,),  # 0...C-1
     'features': List[Tensor (D,)]}  # D=512
    """

    def __init__(self, frcnn_module: torch.nn.Module, out_dim: int = 512):
        super().__init__()
        self.frcnn = frcnn_module
        self.core = getattr(frcnn_module, "model", frcnn_module)
        self.out_dim = out_dim
        self._box_head_out = 1024
        self.projection_head = (
            torch.nn.Identity()
            if self._box_head_out == out_dim
            else torch.nn.Linear(self._box_head_out, out_dim)
        )

    @torch.no_grad()
    def forward(self, images: List[Tensor]) -> List[Dict[str, Tensor]]:
        self.core.eval()  # Ensure the model is in eval mode
        device = next(self.core.parameters()).device
        preds = self.core([image.to(device) for image in images])

        image_list, _ = self.core.transform([image.to(device) for image in images])

        resized_sizes = image_list.image_sizes
        batch = image_list.tensors

        features = self.core.backbone(batch)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        original_sizes = [(image.shape[-2], image.shape[-1]) for image in images]
        boxes_resized: List[Tensor] = []

        for p, osz, rsz in zip(preds, original_sizes, resized_sizes):
            boxes = p["boxes"]
            if boxes.numel() > 0:
                boxes_resized.append(_rescale_boxes_to_resized(boxes, osz, rsz))
            else:
                boxes_resized.append(boxes)
        roi_pool: MultiScaleRoIAlign = self.core.roi_heads.box_roi_pool
        pooled_features = roi_pool(features, boxes_resized, image_list.image_sizes)
        box_features = self.core.roi_heads.box_head(pooled_features)
        if self.out_dim != self._box_head_out:
            box_features = self.projection_head(box_features)
        else:
            box_features = box_features.view(box_features.size(0), -1)

        output: List[Dict[str, Tensor]] = []
        offset = 0
        for p in preds:
            n = p["boxes"].shape[0]
            f = box_features[offset : offset + n]
            offset += n
            output.append(
                {
                    "boxes": p["boxes"],
                    "scores": p["scores"],
                    "labels": p["labels"] - 1,  # Shift from 1...C to 0...C-1
                    "features": f,
                }
            )
        return output
