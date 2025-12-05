from __future__ import annotations
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as T
from timm.data import resolve_data_config

from backbones import get_backbone
from segmenters import BaseSegmenter


class PCASegmenter(BaseSegmenter):
    def __init__(
        self,
        backbone_name: str = "dinov3_base",
        device: str | None = None,
        threshold: float = 2.5,
        kernel_size: int = 5,
        border: float = 0.2,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = get_backbone(backbone_name).to(self.device)
        self.model.eval()


        cfg = resolve_data_config({}, model=self.model)
        _, img_size, _ = cfg["input_size"]
        arch = getattr(getattr(self.model, "pretrained_cfg", {}), "get", lambda k, d=None: {})(  # type: ignore[arg-type]
            "architecture", ""
        )
        if isinstance(arch, str) and "dinov3" in arch:
            img_size = max(img_size, 512)

        self.img_size = img_size
        interp = cfg.get("interpolation", "bicubic")
        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size), interpolation=getattr(T.InterpolationMode, interp.upper(), T.InterpolationMode.BICUBIC)),
                T.ToTensor(),
                T.Normalize(mean=cfg.get("mean", (0.485, 0.456, 0.406)), std=cfg.get("std", (0.229, 0.224, 0.225))),
            ]
        )
        self.threshold = threshold
        self.border = border
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    def get_object_mask(self, image: np.ndarray) -> np.ndarray:
        h0, w0 = image.shape[:2]
        pil = Image.fromarray(image.astype(np.uint8))
        x = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.model.forward_features(x)

        tokens = out.get("x_norm_patchtokens") if isinstance(out, dict) else out

        if tokens is None and isinstance(out, dict):
            tokens = out.get("x")
        if tokens is not None and tokens.ndim == 4:
            B, C, Hf, Wf = tokens.shape
            tokens = tokens.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)

        gh_dyn = int(np.sqrt(tokens.shape[1]))
        gw_dyn = max(1, tokens.shape[1] // max(1, gh_dyn))
        gh, gw = gh_dyn, gw_dyn

        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            gh0, gw0 = self.model.patch_embed.grid_size
            if gh0 * gw0 == tokens.shape[1]:
                gh, gw = gh0, gw0
        n_patches = gh * gw
        tokens = tokens[:, -n_patches:, :]

        feats = tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)
        feats -= feats.mean(0, keepdims=True)
        u, s, vh = np.linalg.svd(feats, full_matrices=False)
        pc1 = vh[0]
        scores = feats @ pc1
        mask = scores > self.threshold
        m_grid = mask.reshape(gh, gw)
        bh = int(gh * self.border)
        bw = int(gw * self.border)
        inner = m_grid[bh : gh - bh, bw : gw - bw]
        if inner.size > 0 and inner.mean() <= 0.35:
            mask = scores < -self.threshold
            m_grid = mask.reshape(gh, gw)
        mask = m_grid.astype(np.uint8)
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)
