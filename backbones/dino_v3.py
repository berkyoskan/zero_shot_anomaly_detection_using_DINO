import timm
"""
DinoV3 backbones
"""

def build_dinov3_small(**kwargs):
    model = timm.create_model(
        "vit_small_plus_patch16_dinov3.lvd1689m",
        pretrained=True,
        num_classes=0,
        **kwargs,
    )
    return model


def build_dinov3_base(**kwargs):
    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m",
        pretrained=True,
        num_classes=0,
        **kwargs,
    )
    return model


def build_dinov3_large(**kwargs):
    model = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m",
        pretrained=True,
        num_classes=0,
        **kwargs,
    )
    return model
