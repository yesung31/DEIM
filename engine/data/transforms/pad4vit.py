import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps

from ...core import register


@register()
class Pad4ViT(T.Transform):
    """
    Symmetrically pad the input tensor to a multiple of patch_size.
    """

    def __init__(self, patch_size=14, pad_value=0):
        super().__init__()
        self.patch_size = patch_size
        self.pad_value = pad_value

    def forward(self, sample):
        raise NotImplementedError("Target coordinate change is not implemented.")
        # Get the size of the image: (H, W)
        image, target, dataset = sample
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Calculate total padding needed
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        # Split padding equally to both sides
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding (pad order: left, right, top, bottom)
        if isinstance(image, Image.Image):
            padded_image = ImageOps.expand(
                image,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=self.pad_value,
            )
        elif isinstance(image, torch.Tensor):
            padded_image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=self.pad_value,
            )
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return padded_image, target, dataset
