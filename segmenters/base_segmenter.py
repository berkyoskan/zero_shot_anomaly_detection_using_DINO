
import numpy as np

class BaseSegmenter:
    """Base class for segmentation models """
    def get_object_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image
        Returns:
            bool mask of shape, where True = foreground object.
        """
        raise NotImplementedError

