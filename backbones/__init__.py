import timm

from backbones.dino_v2 import build_dinov2_small, build_dinov2_base, build_dinov2_large
from backbones.dino_v3 import build_dinov3_small, build_dinov3_base, build_dinov3_large

"""
Model registry for backbones
"""


_BACKBONES = {
    "dinov2_small": build_dinov2_small,
    "dinov2_base": build_dinov2_base,
    "dinov2_large": build_dinov2_large,
    "dinov3_small": build_dinov3_small,
    "dinov3_base": build_dinov3_base,
    "dinov3_large": build_dinov3_large,

}

def get_backbone(name: str, **kwargs):
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(_BACKBONES)}")
    return _BACKBONES[name](**kwargs)
