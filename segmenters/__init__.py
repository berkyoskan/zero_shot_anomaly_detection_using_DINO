from segmenters.base_segmenter import BaseSegmenter
from segmenters.sam3 import SAM3Segmenter


"""
Registry patter style - when you add new segmenter - add it here together with the name
"""
_SEGMENTERS = {
    "sam3": SAM3Segmenter,
}

def get_segmenter(name: str, **kwargs):
    if name not in _SEGMENTERS:
        raise ValueError(f"Unknown segmenter '{name}'. Available: {list(_SEGMENTERS)}")
    return _SEGMENTERS[name](**kwargs)