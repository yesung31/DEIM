import torch
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps

from ...core import register


@register()
class TopCrop(T.Transform):
    """
    Symmetrically pad the input tensor to a multiple of patch_size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, sample):
        # Get the size of the image: (H, W)
        image, target, dataset = sample

        # Get dimensions
        width, height = image.size
        # Calculate crop dimensions
        crop_height = height // 2
        # Crop top half using PIL crop: (left, top, right, bottom)
        image = image.crop((0, 0, width, crop_height))

        return image, target, dataset
