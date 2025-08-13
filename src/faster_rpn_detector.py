import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Tuple
from torch import Tensor

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.models.detection.roi_heads import fastrcnn_loss


class ObjectModule(nn.Module):
    def __init__(self, num_classes, score_thresh_test: float = 0.05):
        super().__init__()
        self.num_classes = int(num_classes)
        self.model = self._create_faster_rcnn_model()
        self.score_thresh_test = float(score_thresh_test)

    def _create_faster_rcnn_model(self):
        """
        Builds the base Faster R-CNN model with a ResNet-50-FPN backbone
        and overrides its box predictor to match the desired number of classes.
        """
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        model.rpn.post_nms_top_n_train = 2000
        model.rpn.post_nms_top_n_test = 1000

        # Override the final classification layers
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes + 1
        )
        model.roi_heads.score_thresh = 0.0
        return model

    def forward(self, images, targets: Optional[List[Dict[str, Tensor]]] = None):
        """
        Standard forward pass for detection:
          - If 'targets' is given, returns a dict of losses {loss_classifier, loss_box_reg, ...}
          - Otherwise, returns a list of detection dicts with 'boxes','labels','scores'.
        """
        if targets is not None:
            tv_targets = [
                {"boxes": t["boxes"], "labels": t["labels"] + 1} for t in targets
            ]
            return self.model(images, tv_targets)
        preds = self.model(images)
        for p in preds:
            if "labels" in p:
                p["labels"] = p["labels"].clamp_min(1) - 1
        return preds

    @torch.no_grad()
    def extract_roi_and_loss(
        self,
        images,
        targets: Optional[List[Dict[str, Tensor]]],
        train_mode: bool = True,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Train mode:
            - computes standard detection losses with given targets
            - Returns per-image dicts with ROI features pooled at GT boxes:
                {boxes, labels (0...C-1), scores(=1), roi_features, image_size}
        Eval mode:
            - runs detector (RPN + ROI, NMS) with score_thresh_test
            - returns per image dicts pooled at final detection boxes
            {boxes, labels (0...C-1), scores, roi_features, image_size}
        """
        if train_mode and targets is None:
            raise ValueError("[ERROR] targets must be provided in training mode.")
        device = images[0].device

        tv_targets: Optional[list[Dict[str, Tensor]]] = None
        if targets is not None:
            if isinstance(targets, list):
                tv_targets = [
                    {"boxes": t["boxes"], "labels": t["labels"] + 1} for t in targets
                ]

            elif isinstance(targets, dict):
                tv_targets = []
                B = len(targets["boxes"])
                lab_key = "labels" if "labels" in targets else "classes"
                for i in range(B):
                    tv_targets.append(
                        {
                            "boxes": targets["boxes"][i],
                            "labels": targets[lab_key][i] + 1,
                        }
                    )
            else:
                raise TypeError(
                    f"[ERROR] Unsupported targets type: {type(targets)}. "
                    "Expected list[dict] or dict with 'boxes' and 'labels'."
                )
        if train_mode == True:
            self.model.train()
        else:
            self.model.eval()

        transformed = (
            self.model.transform(images, tv_targets)
            if tv_targets is not None
            else self.model.transform(images)
        )
        image_list = transformed[0]  # Extract ImageList

        # Extract features from the backbone
        features = self.model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        image_sizes = image_list.image_sizes

        if train_mode:
            proposals, rpn_losses = self.model.rpn(image_list, features, tv_targets)
            for idx, (prop, tgt) in enumerate(zip(proposals, tv_targets)):
                gt = tgt["boxes"].to(prop.device)
                if gt.numel():
                    proposals[idx] = torch.cat(
                        [prop, gt], dim=0
                    )  # Append ground truth boxes
        else:
            proposals, rpn_losses = self.model.rpn(image_list, features)

        # Get image sizes

        # ROI Processing
        box_roi_pool = self.model.roi_heads.box_roi_pool
        box_head = self.model.roi_heads.box_head
        box_predictor = self.model.roi_heads.box_predictor

        pooled = box_roi_pool(features, proposals, image_sizes)
        box_head_features = box_head(pooled)

        class_logits, box_regression = box_predictor(box_head_features)

        roi_losses: Dict[str, Tensor] = {}
        if train_mode:
            # Pass targets_list to the ROI sample selection as well
            proposals, matched_idxs, labels, regression_targets = (
                self.model.roi_heads.select_training_samples(proposals, tv_targets)
            )
            pooled_2 = box_roi_pool(features, proposals, image_list.image_sizes)
            box_head_features_2 = box_head(pooled_2)
            class_logits_2, box_regression_2 = box_predictor(box_head_features_2)

            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits_2, box_regression_2, labels, regression_targets
            )

            roi_losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }

        detection_losses: Dict[str, Tensor] = {}
        detection_losses.update(rpn_losses)
        detection_losses.update(roi_losses)

        results: List[Dict[str, Tensor]] = []

        if train_mode:
            # Pool ROI features at GT boxes (stable supervision)
            C = self.model.roi_heads.box_head.fc6.out_features
            for i, t in enumerate(tv_targets or []):
                gt_boxes = t["boxes"]
                if gt_boxes.numel() == 0:
                    results.append(
                        {
                            "boxes": torch.empty((0, 4), device=device),
                            "labels": torch.empty(
                                (0,), dtype=torch.long, device=device
                            ),
                            "scores": torch.empty((0,), device=device),
                            "roi_features": torch.empty((0, C), device=device),
                            "image_size": torch.tensor(image_sizes[i], device=device),
                        }
                    )
                    continue
                pooled_gt = box_roi_pool(features, [gt_boxes], [image_sizes[i]])
                feats_gt = box_head(pooled_gt)  # (G, C)
                results.append(
                    {
                        "boxes": gt_boxes.to(device),
                        "labels": (t["labels"] - 1).to(device),  # back to 0..C-1
                        "scores": torch.ones((gt_boxes.size(0),), device=device),
                        "roi_features": feats_gt,
                        "image_size": torch.tensor(image_sizes[i], device=device),
                    }
                )
            return detection_losses, results

        old_thresh = float(self.model.roi_heads.score_thresh)
        self.model.roi_heads.score_thresh = self.score_thresh_test
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(
            class_logits, box_regression, proposals, image_sizes
        )
        self.model.roi_heads.score_thresh = old_thresh

        for i, (b_i, s_i, l_i) in enumerate(zip(boxes, scores, labels)):
            if b_i.numel() == 0:
                C = self.model.roi_heads.box_head.fc6.out_features
                results.append(
                    {
                        "boxes": b_i.to(device),
                        "labels": (l_i.clamp_min(1) - 1).to(device),
                        "scores": s_i.to(device),
                        "roi_features": torch.empty((0, C), device=device),
                        "image_size": torch.tensor(image_sizes[i], device=device),
                    }
                )
                continue
            pooled_det = box_roi_pool(features, [b_i], [image_sizes[i]])
            feats_det = box_head(pooled_det)  # (D, C)
            results.append(
                {
                    "boxes": b_i.to(device),
                    "labels": (l_i.clamp_min(1) - 1).to(device),
                    "scores": s_i.to(device),
                    "roi_features": feats_det,
                    "image_size": torch.tensor(image_sizes[i], device=device),
                }
            )

        return {}, results
