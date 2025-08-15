from typing import Any, Dict, List

import numpy as np
import torch
from PIL.Image import Image


def upscale_image(img, upscale_model, outscale=2):
    """
    Upscale an image using RealESRGAN.

    Args:
        img: A cv2 image (numpy array)
        model_path: Path to the model weights
        outscale: The output scale (default: 4)

    Returns:
        Enhanced image
    """
    if isinstance(img, Image):
        img = np.array(img)
    with torch.no_grad():
        output, _ = upscale_model.enhance(img, outscale=outscale)
    return output


def upscale_images(images: List[np.ndarray], scale_factor: int = 3) -> List[np.ndarray]:
    """Upscale images using the specified scale factor."""
    upscaled_images = []
    for img in images:
        upscaled_img = upscale_image(img, outscale=scale_factor)
        upscaled_images.append(upscaled_img)
    return upscaled_images
