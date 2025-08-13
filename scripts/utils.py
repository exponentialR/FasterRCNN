import numpy as np
import json
import torch
from torch import Tensor
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Union


def compute_dataset_translation_norm(dataset_jsons_path):
    """
    Computes the mean translation vector from a dataset of JSON files containing 6DOF poses.
    :param dataset_jsons_path:
    :return:
    """
    all_translations = []
    for json_files in dataset_jsons_path.glob("*.json"):
        json_data = json.loads(json_files.read_text())
        for obj in json_data["objects"]:
            all_translations.append(obj["6DOF_pose"]["position"])
    all_translations = np.array(all_translations)
    mean_translation = np.mean(all_translations, axis=0)
    standard_deviation_translation = np.std(all_translations, axis=0)
    print(f"Mean translation: {mean_translation}")
    print(f"Standard deviation of translation: {standard_deviation_translation}")

    return mean_translation, standard_deviation_translation


def count_parameters(module):
    """
    Count the number of trainable parameters in a PyTorch module.
    :param module: PyTorch module
    :return: Number of trainable parameters
    """
    total_parameters = sum(p.numel() for p in module.parameters())
    trainable_parameters = sum(
        p.numel() for p in module.parameters() if p.requires_grad
    )
    return total_parameters, trainable_parameters


def normalise_and_print(dataset_jsons_path, mean_translation, std_translation):
    """
    Normalises the translations in the dataset JSON files and prints them.
    :param dataset_jsons_path:
    :param mean_translation:
    :param std_translation:
    :return:
    """
    for json_files in dataset_jsons_path.glob("*.json"):
        json_data = json.loads(json_files.read_text())
        for obj in json_data["objects"]:
            position = np.array(obj["6DOF_pose"]["position"])
            normalised_position = (position - mean_translation) / std_translation
            print(f"Normalised position: {normalised_position}")


def rot6d_to_rotmat(x: Tensor) -> Tensor:
    b1 = F.normalize(x[:, 0:3], dim=1)
    proj = (b1 * x[:, 3:6]).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(x[:, 3:6] - proj, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def quaternion_to_rotmat(q: Tensor) -> Tensor:  # (w,x,y,z)
    w, x, y, z = q.unbind(1)
    B = q.size(0)
    R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_canon_key(dir_path: Path) -> Dict[int, torch.Tensor]:
    mapping: Dict[int, torch.Tensor] = {}
    for fp in dir_path.glob("*.json"):
        data = json.loads(fp.read_text())
        rows = [
            [kv["x"], kv["y"], kv["z"]] for _, kv in sorted(data["keypoints"].items())
        ]
        mapping[fp.stem.lower()] = torch.tensor(rows, dtype=torch.float32)
    if not mapping:
        raise RuntimeError("No canonical key‑points found in " + str(dir_path))
    return mapping


def load_canonical_keypoints(dir_path: Path) -> Dict[str, torch.Tensor]:
    """Return mapping ``class_name.lower() -> (21,3) float32 Tensor``."""
    mapping: Dict[str, torch.Tensor] = {}
    for fp in sorted(dir_path.glob("*.json")):
        data = json.loads(fp.read_text())
        kps: Dict[str, Dict[str, float]] = data.get("keypoints", {})
        # Sort by key name (k11, k12, ... k31) to enforce consistent order
        rows: List[List[float]] = [
            [kv["x"], kv["y"], kv["z"]] for _, kv in sorted(kps.items())
        ]
        mapping[fp.stem.lower()] = torch.tensor(rows, dtype=torch.float32)
    if not mapping:
        raise RuntimeError(f"No canonical key‑point JSONs found in {dir_path}")
    return mapping


# Quaternion conversion helper (matrix→wxyz)
def rotmat_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(N,3,3) → (N,4) wxyz."""
    w = torch.sqrt(torch.clamp(R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1, min=1e-6)) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w + 1e-6)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w + 1e-6)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w + 1e-6)
    return torch.stack([w, x, y, z], dim=1)


def reorthonormalise_so3(R: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(R)
    UV = U @ Vh
    detUV = torch.det(UV)
    D = torch.diag_embed(
        torch.stack(
            [torch.ones_like(detUV), torch.ones_like(detUV), detUV.sign()], dim=-1
        )
    )
    return U @ D @ Vh


if __name__ == "__main__":
    from pathlib import Path

    dataset_jsons_path = Path("/home/samuel/Downloads/small_set/compact")
    mean_translation, std_translation = compute_dataset_translation_norm(
        dataset_jsons_path
    )
    normalise_and_print(dataset_jsons_path, mean_translation, std_translation)
