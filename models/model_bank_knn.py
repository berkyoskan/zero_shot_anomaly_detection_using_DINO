from __future__ import annotations
from typing import Iterable, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from segmenters import BaseSegmenter
from utils.visualize import visualize_segmentation


class PatchKNNDetector:
    """
    AnomalyDINO style model:

    1) Using backbone (for now only DINOv2) to get patch embeddings.
    2) Using a segmenter (for now only SAM2) to keep only foreground patches.
    3) Building a memory bank from few normal reference images
    4) For a test image, computes kNN distance of each patch to the memory bank
    5) Return an anomaly heatmap and an image-level score.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        segmenter: Optional[BaseSegmenter] = None,
        device: str = "cuda",
        k_neighbors: int = 1,
    ) -> None:
        self.device = device
        self.backbone = backbone.to(device)
        self.backbone.eval()

        self.segmenter = segmenter
        self.k_neighbors = k_neighbors

        # Prepare timm transforms for this backbone. :contentReference[oaicite:5]{index=5}
        data_cfg = resolve_data_config({}, model=self.backbone)
        self.transform = create_transform(**data_cfg)

        self.num_register_tokens = getattr(self.backbone, "num_register_tokens", 0)

        # Memory bank of foreground patch embeddings
        self.memory_bank: Optional[np.ndarray] = None
        self.patch_grid_size: Optional[Tuple[int, int]] = None


    def fit(
        self,
        train_image_paths: Iterable[str],
        n_ref: int = 1,
    ) -> None:
        """
        Build memory bank from normal images.
        """

        selected_paths = list(train_image_paths)[:n_ref]
        all_patches: list[np.ndarray] = []

        for path in selected_paths:
            #Extracting features
            image = self._load_image(path)
            patch_feats, grid_size = self._extract_patch_features(image)

            # Applying oreground mask
            patch_mask = self._compute_patch_mask(image, grid_size)
            patch_feats_fg = patch_feats[patch_mask]

            all_patches.append(patch_feats_fg)
            self.patch_grid_size = grid_size

        self.memory_bank = np.concatenate(all_patches, axis=0)

    def predict(
        self,
        image_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run anomaly detection on a single test image.

        Returns:
            image_rgb:
            anomaly_map
            image_score
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Call `fit(...)` first.")

        image = self._load_image(image_path)
        patch_feats, grid_size = self._extract_patch_features(image)

        patch_mask = self._compute_patch_mask(image, grid_size)

        # Compute distances only on foreground patches
        scores_fg = self._knn_distances(patch_feats[patch_mask])

        # Put scores back into full patch grid
        scores_all = np.zeros(patch_feats.shape[0], dtype=np.float32)
        scores_all[patch_mask] = scores_fg
        patch_map = scores_all.reshape(grid_size)

        # Upsample to full image size for visualization
        h, w = image.shape[:2]
        anomaly_map = cv2.resize(
            patch_map,
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)

        # Simple image-level score: mean of top 1% patch scores
        image_score = self._mean_top_percent(scores_fg, top_percent=1.0)

        return image, anomaly_map, image_score

    #This part was heavily written using ChatGPT -
    # expect a lot of bullshit and need to check if everything is correctly implemented

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load an image in RGB format."""
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    @staticmethod
    def _l2_normalize(feats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norm, eps)

    def _extract_patch_features(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Run backbone on a single image and return:
            - patch features
            - grid_size
        """
        pil = Image.fromarray(image)
        x = self.transform(pil).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.inference_mode():
            out = self.backbone.forward_features(x)

        if isinstance(out, dict):
            # Newer timm ViTs sometimes return a dict; prefer patch tokens if present

            if "x_norm_patchtokens" in out:
                tokens = out["x_norm_patchtokens"]  # (B, N_patches, C)
            elif "x" in out:
                tokens = out["x"]  # fallback
            else:
                raise ValueError(
                    f"Dict features from backbone missing 'x_norm_patchtokens'/'x' keys: {out.keys()}"
                )
        elif isinstance(out, torch.Tensor):
            if out.ndim == 3:
                # ViT / DINOv2 case: (B, N_tokens, C)
                tokens = out
            elif out.ndim == 4:
                # CNN-style case: (B, C, Hf, Wf)
                B, C, Hf, Wf = out.shape
                tokens = out.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)  # -> (B, N, C)
            else:
                raise ValueError(
                    f"Unsupported tensor shape from backbone: {tuple(out.shape)}"
                )
        else:
            raise ValueError(f"Unsupported features from backbone type: {type(out)}")

        # --- 2) Separate patch tokens & get grid size ----------------------
        B, N, C = tokens.shape

        # For ViTs with patches, timm exposes the patch grid size here.
        # For DINOv2 reg4 on 518x518, this will be (37, 37). :contentReference[oaicite:1]{index=1}
        if hasattr(self.backbone, "patch_embed") and hasattr(
            self.backbone.patch_embed, "grid_size"
        ):
            gh, gw = self.backbone.patch_embed.grid_size  # e.g. (37, 37)
            n_patches = gh * gw
        else:
            # Fallback: assume tokens are already only patches and form a square grid.
            n_patches = N
            gh = int(np.sqrt(n_patches))
            gw = n_patches // gh if gh > 0 else n_patches

        if N == n_patches:
            # tokens already only patches
            patch_tokens = tokens
        elif N > n_patches:
            # Typical ViT case: prefix tokens (CLS + registers) then spatial tokens.
            # For DINOv2 reg4: 1 cls + 4 registers + 37*37 patches = 5 + 1369 = 1374.
            # We just take the last n_patches tokens as patches.
            patch_tokens = tokens[:, -n_patches:, :]
        else:
            raise ValueError(
                f"Not enough tokens ({N}) to fill patch grid ({n_patches})."
            )

        # --- 3) Flatten and L2-normalize -----------------------------------
        feats = (
            patch_tokens.reshape(B * n_patches, C)
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )
        feats = self._l2_normalize(feats)  # for cosine-like distances

        grid_size = (gh, gw)
        return feats, grid_size

    def _compute_patch_mask(
        self,
        image: np.ndarray,
        grid_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Convert a pixel-level mask to patch-level mask.

        If no segmenter is provided, returns all‑True mask.
        """
        h_p, w_p = grid_size
        n_patches = h_p * w_p

        if self.segmenter is None:
            return np.ones(n_patches, dtype=bool)

        full_mask = self.segmenter.get_object_mask(image)  # (H, W) bool
        visualize_segmentation(
            image,
            full_mask,
            grid_size=grid_size,  # optional
            title=f"Segmentation debug:",
        )
        full_mask_uint8 = full_mask.astype(np.uint8)

        # Downsample to patch grid (nearest neighbor to keep binary)
        mask_small = cv2.resize(
            full_mask_uint8,
            (w_p, h_p),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        return mask_small.reshape(-1)

    def _knn_distances(self, feats: np.ndarray) -> np.ndarray:
        """
        Compute distance of each query feature to its nearest neighbors in the memory bank.

        Very simple NumPy implementation: fine for few-shot memory sizes.
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty.")

        # Both feats and memory_bank are L2-normalized, so
        # ||a - b||^2 = 2 - 2 * cos(a, b). We can either use dot-product or full L2.
        # Here we just do full L2 distance for clarity (still fast at this scale).
        a = feats  # (N, C)
        b = self.memory_bank  # (M, C)

        # Compute squared L2 distances in a vectorized way
        a2 = np.sum(a**2, axis=1, keepdims=True)        # (N, 1)
        b2 = np.sum(b**2, axis=1, keepdims=True).T      # (1, M)
        ab = a @ b.T                                    # (N, M)

        d2 = np.clip(a2 + b2 - 2.0 * ab, a_min=0.0, a_max=None)
        d = np.sqrt(d2)                                 # (N, M)

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
        """Mean of top p% values (default 1%) as simple image-level anomaly score."""
        if values.size == 0:
            return 0.0
        k = max(1, int(round(values.size * (top_percent / 100.0))))
        # Top-k (largest)
        part = np.partition(values, -k)[-k:]
        return float(part.mean())
