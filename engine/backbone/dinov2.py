import torch
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
        local_model_dir="weight/dinov2/",
    ):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        name = name.lower()
        name = f"dinov2_vit{name}14_reg"

        if pretrained:
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    self.network = torch.hub.load("facebookresearch/dinov2", name).to(
                        "cpu"
                    )
                    torch.distributed.barrier()
                else:
                    torch.distributed.barrier()
                    self.network = torch.hub.load("facebookresearch/dinov2", name).to(
                        "cpu"
                    )
            else:
                self.network = torch.hub.load("facebookresearch/dinov2", name).to("cpu")
        else:
            raise NotImplementedError(
                "DINOv2 IS a pretrained model, please set pretrained=True"
            )

        self.return_idx = sorted(
            [
                idx if idx >= 0 else len(self.network.blocks) + idx
                for idx in self.return_idx
            ]
        )

        if freeze_at >= 0:
            self._freeze_parameters(self.network.patch_embed)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.blocks))):
                    self._freeze_parameters(self.blocks[i])

        if freeze_norm:
            self._freeze_norm(self.network)

    def _freeze_norm(self, module):
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = False
        else:
            for name, child in module.named_children():
                self._freeze_norm(child)

    def _freeze_parameters(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.network.patch_embed(x)
        outs = []
        for idx, block in enumerate(self.network.blocks):
            x = block(x)
            if idx == len(self.network.blocks) - 1:
                x = self.network.norm(x)
                x = self.network.head(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
