
import timm
"""
DinoV2 backbones
"""
def build_dinov2_small(**kwargs):

    model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m',
                              pretrained=True, num_classes=0)
    return model

def build_dinov2_base(**kwargs):

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m',
                              pretrained=True, num_classes=0)
    return model

def build_dinov2_large(**kwargs):

    model = timm.create_model('vit_large_patch14_dinov2.lvd142m',
                              pretrained=True, num_classes=0)
    return model