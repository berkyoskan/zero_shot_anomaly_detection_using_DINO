from __future__ import annotations
from typing import Iterable, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from timm.data import resolve_data_config
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from segmenters import BaseSegmenter
from utils.visualize import visualize_segmentation


class PatchKNNDetector:
    """
    Model bank (Anomaly Dino) style model :
    """

    def __init__(self, backbone, segmenter = None, device = "cuda", k_neighbors = 1):


        self.device = device
        self.backbone = backbone.to(device)
        #Switch backbone to inference mode 
        self.backbone.eval()

        self.segmenter = segmenter
        self.k_neighbors = k_neighbors

        # Prepare resize/normalize augmentations shared by DINO and SAM
        data_cfg = resolve_data_config({}, model=self.backbone)
        _, self.img_size, _ = data_cfg["input_size"]
        interp = data_cfg.get("interpolation", "bicubic")

        self.transform = T.Compose(
            [
                T.Resize(self.img_size, interpolation=getattr(InterpolationMode, interp.upper(), InterpolationMode.BICUBIC)),
                T.ToTensor(),
                T.Normalize(mean=data_cfg.get("mean", (0.485, 0.456, 0.406)),
                            std=data_cfg.get("std", (0.229, 0.224, 0.225))),
            ]
        )

        self.num_register_tokens = getattr(self.backbone, "num_register_tokens", 0)

        # Memory bank of foreground patch embeddings
        self.memory_bank = None
        self.patch_grid_size = None


    def fit(self, train_image_paths, n_ref = 1):
        """Populate memory bank with references """

        selected_paths = list(train_image_paths)[:n_ref]
        all_patches = []

        for path in selected_paths:

            #Extracting features
            image = self._load_image(path)
            patch_feats, grid_size = self._extract_patch_features(image)

            # Applying foreground mask
            patch_mask = self._compute_patch_mask(image, grid_size)
            patch_feats_fg = patch_feats[patch_mask]

            all_patches.append(patch_feats_fg)
            self.patch_grid_size = grid_size

        self.memory_bank = np.concatenate(all_patches, axis=0)

    def predict(self, image_path) :
        """
        Run anomaly detection inference
        """
        image = self._load_image(image_path)
        patch_feats, grid_size = self._extract_patch_features(image)

        patch_mask = self._compute_patch_mask(image, grid_size)

        # Compute distances only on foreground patches
        scores_fg = self._knn_distances(patch_feats[patch_mask])

        # Put scores back into full patch grid
        scores_all = np.zeros(patch_feats.shape[0], dtype=np.float32)
        scores_all[patch_mask] = scores_fg
        patch_map = scores_all.reshape(grid_size)

        # Upsample to full image ( just for visualization)
        h, w = image.shape[:2]
        anomaly_map = cv2.resize(
            patch_map,
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)

        # Using image-level score - mean of top 1% patch scores
        image_score = self._mean_top_percent(scores_fg, top_percent=1.0)

        return image, anomaly_map, image_score


    def _load_image(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    @staticmethod
    def _l2_normalize(feats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norm, eps)

    def _extract_patch_features(self, image: np.ndarray) :
        """
        Run backbone on a single image and return patch features
        """
        pil_resized, _ = self._resize_for_model(image)
        x = self.transform(pil_resized).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.backbone.forward_features(x)

        tokens = out.get("x_norm_patchtokens") if isinstance(out, dict) else out
        
        if tokens is None and isinstance(out, dict):
            tokens = out.get("x")
        if tokens is not None and tokens.ndim == 4:
            B, C, Hf, Wf = tokens.shape
            tokens = tokens.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)

        B, N, C = tokens.shape

        if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "grid_size"):
            gh, gw = self.backbone.patch_embed.grid_size
        else:
            gh = int(np.sqrt(N))
            gw = max(1, N // max(1, gh))

        n_patches = gh * gw
        patch_tokens = tokens[:, -n_patches:, :]

        # Flatten and normalize
        feats = (
            patch_tokens.reshape(B * n_patches, C)
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )
        feats = self._l2_normalize(feats)

        grid_size = (gh, gw)
        return feats, grid_size

    def _compute_patch_mask(self,image,grid_size) :
        """
        Convert a pixel-level mask to patch-level mask.
        """
        h_p, w_p = grid_size
        n_patches = h_p * w_p

        if self.segmenter is None:
            return np.ones(n_patches, dtype=bool)

        # Resize image same way as in the backbone before sending to SAM
        pil_resized, np_resized = self._resize_for_model(image)
        full_mask = self.segmenter.get_object_mask(np_resized) 

        # Optionally visualize in resized space
        visualize_segmentation(
            np_resized,
            full_mask,
            grid_size=None,
            title=f"Segmentation debug (resized {self.img_size})",
        )

        full_mask_uint8 = (full_mask.astype(np.uint8) * 255).astype(np.float32)

        # Downsample to patch grid with area interpolation for coverage
        mask_small = cv2.resize(
            full_mask_uint8,
            (w_p, h_p),
            interpolation=cv2.INTER_AREA,
        ) / 255.0

        patch_mask = (mask_small >= 0.5).reshape(-1)

        # Fallback if mask collapses
        fg_ratio = patch_mask.mean()
        if fg_ratio < 0.01 or fg_ratio > 0.99:
            patch_mask = np.ones(n_patches, dtype=bool)

        return patch_mask

    def _resize_for_model(self, image):
        pil = Image.fromarray(image)
        pil_resized = pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        np_resized = np.array(pil_resized)
        return pil_resized, np_resized

    def _knn_distances(self, feats: np.ndarray) -> np.ndarray:
        """
        Compute distance of each query feature to its nearest neighbors in the memory bank.

        Very simple NumPy implementation: fine for few-shot memory sizes.
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty.")

        
        a = feats 
        b = self.memory_bank 

        # vectorize version of L2 distances 
        a2 = np.sum(a**2, axis=1, keepdims=True)        
        b2 = np.sum(b**2, axis=1, keepdims=True).T    
        ab = a @ b.T                                   
        # Clip to avoid negative values
        d2 = np.clip(a2 + b2 - 2.0 * ab, a_min=0.0, a_max=None)
        d = np.sqrt(d2) 

        # kNN: take mean of k smallest distances per patch
        k = min(self.k_neighbors, d.shape[1])
        if k == 1:
            min_d = d.min(axis=1)
        else:
            # partial sort for efficiency
            part = np.partition(d, kth=k - 1, axis=1)[:, :k]
            min_d = part.mean(axis=1)

        return min_d.astype(np.float32)

    @staticmethod
    def _mean_top_percent(values: np.ndarray, top_percent: float = 1.0) -> float:
        """Mean of top p% values used as image level anomaly score."""
        if values.size == 0:
            return 0.0
        k = max(1, int(round(values.size * (top_percent / 100.0))))
        part = np.partition(values, -k)[-k:]
        return float(part.mean())
