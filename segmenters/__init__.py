from segmenters.base_segmenter import BaseSegmenter


from segmenters.sam3 import SAM3Segmenter

from segmenters.pca_segmenter import PCASegmenter

"""
Model registry for segmenters
"""
_SEGMENTERS = {}
if SAM3Segmenter is not None:
    _SEGMENTERS["sam3"] = SAM3Segmenter
if PCASegmenter is not None:
    _SEGMENTERS["pca"] = PCASegmenter


def get_segmenter(name: str, **kwargs):
    if name not in _SEGMENTERS:
        raise ValueError(f"Unknown segmenter '{name}'. Available: {list(_SEGMENTERS)}")
    return _SEGMENTERS[name](**kwargs)
