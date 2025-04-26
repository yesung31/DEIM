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
        return_idx=[-3, -2, -1],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=True,
        local_model_dir="weight/dinov2/",
    ):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx
        assert len(self.return_idx) == 3

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

        embed_dim = self.network.patch_embed.proj.out_channels
        self.up_conv = nn.ConvTranspose2d(
            embed_dim, embed_dim, kernel_size=2, stride=2, dilation=1
        )
        self.down_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2)

        self.return_idx = sorted(
            [
                idx if idx >= 0 else len(self.network.blocks) + idx
                for idx in self.return_idx
            ]
        )

        if freeze_at >= 0:
            self._freeze_parameters(self.network.patch_embed)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.network.blocks))):
                    self._freeze_parameters(self.network.blocks[i])

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
        b, c, h, w = x.shape
        # pad_h = (14 - h % 14) % 14
        # pad_w = (14 - w % 14) % 14
        # if pad_h > 0 or pad_w > 0:
        #     # Distribute the padding equally across all sides
        #     pad_top = pad_h // 2
        #     pad_bottom = pad_h - pad_top
        #     pad_left = pad_w // 2
        #     pad_right = pad_w - pad_left
        #     x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        x = self.network.patch_embed(x)
        outs = []
        for idx, block in enumerate(self.network.blocks):
            x = block(x)
            if idx == len(self.network.blocks) - 1:
                x = self.network.norm(x)
                x = self.network.head(x)
            if idx in self.return_idx:
                if idx == self.return_idx[0]:
                    feat = x.transpose(1, 2).reshape(b, -1, h // 14, w // 14)
                    feat = self.up_conv(feat)
                    outs.append(feat)
                elif idx == self.return_idx[1]:
                    feat = x.transpose(1, 2).reshape(b, -1, h // 14, w // 14)
                    outs.append(feat)
                else:  # idx == self.return_idx[2]
                    feat = x.transpose(1, 2).reshape(b, -1, h // 14, w // 14)
                    feat = self.down_conv(feat)
                    outs.append(feat)
        return outs
