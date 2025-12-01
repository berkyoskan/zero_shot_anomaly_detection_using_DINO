
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model

from segmenters import BaseSegmenter


class SAM3Segmenter(BaseSegmenter):
    """
    SAM3 wrapper using a text prompt of object type
    """

    def __init__(
        self,
        text_prompt: str,
        model_name: str = "facebook/sam3",
        device: str = "cuda",
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5 ):
        """
        Args:
            text_prompt: stuff we want to segment.
            model_name: HF repo id for the SAM3 model.
            device: "cuda" or "cpu".
            score_threshold: min detection score to keep an instance.
            mask_threshold: pixel threshold for masks.
        """
        super().__init__()

        if torch.cuda.is_available() and device.startswith("cuda"):
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # preprocess text prompt so metal_nut is processed as metal nut
        self.text_prompt = text_prompt.replace("_", " ")
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

        # Loading model model + defining processor
        self.model = Sam3Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_name)

    def get_object_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Running SAM3 and returning a single foreground mask.
        """
        # Pill image stuff - probably there is less idiotic way, but it is wat ChatGPT suggested
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_image = image

        # defining preprocessor with text prompt
        inputs = self.processor(
            images=pil_image,
            text=self.text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process instance segmentation (HF example style)
        target_sizes = inputs.get("original_sizes").tolist()
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )[0]

        masks = results.get("masks", None)
        scores = results.get("scores", None)

        # If SAM completely fails we keep everything
        if masks is None or masks.numel() == 0:
            H, W = pil_image.size[1], pil_image.size[0] 
            return np.ones((H, W), dtype=bool)

        if scores is not None:
            keep = scores >= self.score_threshold
            if keep.sum() == 0:
                H, W = pil_image.size[1], pil_image.size[0]
                return np.ones((H, W), dtype=bool)
            masks = masks[keep]

        # check if mask passes mask treshold
        masks_bin = (masks > self.mask_threshold)
        combined = masks_bin.any(dim=0)
        full_mask = combined.cpu().numpy().astype(bool)

        return full_mask
