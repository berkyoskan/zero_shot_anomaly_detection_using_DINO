import timm

from backbones.dino_v2 import build_dinov2_small, build_dinov2_base, build_dinov2_large

_BACKBONES = {
    "dinov2_small": build_dinov2_small,
    "dinov2_base": build_dinov2_base,
    "dinov2_large": build_dinov2_large,

}

def get_backbone(name: str, **kwargs):
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(_BACKBONES)}")
    return _BACKBONES[name](**kwargs)