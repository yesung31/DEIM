import torch.nn as nn
from ..core import register


@register()
class DINOv2(nn.Module):
    """
    DINOv2 backbone
    """

    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=True,
        local_model_dir="weight/hgnetv2/",
    ):
        