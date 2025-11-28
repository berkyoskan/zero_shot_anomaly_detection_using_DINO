from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Processor, Sam2Model

from segmenters import BaseSegmenter


class SAM2Segmenter(BaseSegmenter):
    """
    SAM2 wrapper.

    - Uses Sam2Model (e.g. `facebook/sam2.1-hiera-large`).
    - Segments (approximately) all objects in the image by prompting
      with a full-image bounding box and returns a single boolean mask
      given by the union of all predicted masks.
    """

    def __init__(
        self,
        text_prompt: str | None = None,
        model_name: str = "facebook/sam2.1-hiera-large",
        device: str = "cuda",
        mask_threshold: float = 0.5,
    ) -> None:
        """
        Args:
            text_prompt: kept for compatibility with SAM3Segmenter, but unused.
            model_name: HF repo id for the SAM2 model, e.g. "facebook/sam2.1-hiera-large".
            device: "cuda" or "cpu".
            mask_threshold: pixel threshold for masks (after SAM2 post-processing).
        """
        super().__init__()

        if torch.cuda.is_available() and device.startswith("cuda"):
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Load SAM2 model + processor
        self.model = Sam2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = Sam2Processor.from_pretrained(model_name)

    def get_object_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Run SAM2 and return a single foreground mask.

        Current behavior:
        - Convert image to PIL.
        - Use a single bounding box covering the whole image as prompt.
        - Run SAM2, post-process masks to image resolution.
        - Threshold and union all masks into one boolean (H, W) array.

        You can later refine this by:
        - replacing the full-image box with a tighter box around the object of interest
          (e.g., from DINOv3 anomaly heatmap or a detector),
        - or adding point prompts if you have them.
        """
        # Ensure PIL image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_image = image

        W, H = pil_image.size  # PIL: (W, H)

        # Full image bounding box: [x_min, y_min, x_max, y_max]
        input_boxes = [[[0, 0, W, H]]]

        # Build inputs for SAM2
        inputs = self.processor(
            images=pil_image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # multimask_output=False → one mask per box
            outputs = self.model(**inputs, multimask_output=False)

        # Post-process masks to original resolution (HF SAM2 docs style)
        # returns a list over batch; we only have one image, so [0]
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),  # (B, num_masks, H', W')
            inputs["original_sizes"],
        )[0]

        # Shapes can be:
        # - (num_masks, H, W)
        # - or (1, num_masks, H, W) depending on version
        if masks.ndim == 4:
            # (B, num_masks, H, W) -> (num_masks, H, W) for B=1
            masks = masks[0]

        if masks.ndim == 2:
            # Single mask: (H, W)
            full_mask = (masks > self.mask_threshold).numpy().astype(bool)
            return full_mask

        if masks.ndim != 3:
            # Failsafe: if something weird happens, keep everything
            return np.ones((H, W), dtype=bool)

        # masks: (num_masks, H, W)
        masks_bin = masks > self.mask_threshold
        combined = masks_bin.any(dim=0)  # (H, W)
        full_mask = combined.numpy().astype(bool)

        return full_mask
