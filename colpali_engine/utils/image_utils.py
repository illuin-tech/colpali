"""
Utility functions for working with images.
"""

import base64
import io

from PIL import Image


def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    # Calculate the scaling factor
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def scale_to_max_dimension(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """
    Scale an image to a maximum dimension while maintaining the aspect ratio.
    """
    # Get the dimensions of the image
    width, height = image.size

    max_original_dimension = max(width, height)

    if max_original_dimension < max_dimension:
        return image

    # Calculate the scaling factor
    aspect_ratio = max_dimension / max_original_dimension
    new_width = int(width * aspect_ratio)
    new_height = int(height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def get_base64_image(img: str | Image.Image, add_url_prefix: bool = True) -> str:
    """
    Convert an image (from a filepath or a PIL.Image object) to a JPEG-base64 string.
    """
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError("`img` must be a path to an image or a PIL Image object.")

    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")
    b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_data}" if add_url_prefix else b64_data
