from abc import ABC, abstractmethod
from typing import Dict, Tuple

from PIL import Image


class BaseViTConfig(ABC):
    """
    Simplified configuration class for Vision Transformer (ViT) models.
    """

    def __init__(self, patch_size: int):
        self.patch_size = patch_size
        self.resolution = None

    @abstractmethod
    def get_n_patches(self, image: Image.Image) -> Tuple[int, int]:
        """
        Return the number of patches for the input image: (n_patches_x, n_patches_y).
        """
        pass


class VanillaViTConfig(BaseViTConfig):
    """
    Configuration class for the vanilla Vision Transformer (ViT) encoder, i.e. a vision encoder
    that has a fixed patch size and resolution.

    Args:
        patch_size: int
            The size of the square patches extracted from the input image.
        resolution: int, optional
            The resolution of the input image. If not provided, the resolution is dynamic.
    """

    def __init__(
        self,
        patch_size: int,
        resolution: Tuple[int, int],
    ):
        if resolution[0] != resolution[1]:
            raise ValueError("The resolution must correspond to a square image.")

        super().__init__(patch_size)
        self.resolution = resolution

    @property
    def n_patches(self) -> Tuple[int, int]:
        if self.resolution[0] % self.patch_size != 0:
            raise ValueError(f"Resolution {self.resolution} is not divisible by patch size {self.patch_size}")

        n_patches_per_dim = self.resolution[0] // self.patch_size

        return (n_patches_per_dim, n_patches_per_dim)

    def get_n_patches(self, image: Image.Image) -> Tuple[int, int]:
        return self.n_patches


class Qwen2VLViTConfig(BaseViTConfig):
    """
    TODO: Implement the configuration class for the Qwen2VL-ViT model.
    """

    def __init__(self, patch_size: int):
        super().__init__(patch_size)
        raise NotImplementedError("Qwen2VLViTConfig is not implemented yet.")

    def get_n_patches(self, image: Image.Image) -> Tuple[int, int]:
        # NOTE: Need the image to compute the number of patches because Qwen2VL-ViT has a dynamic resolution.
        raise NotImplementedError("Qwen2VLViTConfig is not implemented yet.")


MODEL_NAME_TO_VIT_CONFIG: Dict[str, VanillaViTConfig] = {
    "vidore/colpali": VanillaViTConfig(patch_size=14, resolution=(448, 448)),
    "vidore/colpali-v1.1": VanillaViTConfig(patch_size=14, resolution=(448, 448)),
    "vidore/colpali-v1.2": VanillaViTConfig(patch_size=14, resolution=(448, 448)),
    "vidore/colpali-v1.2-merged": VanillaViTConfig(patch_size=14, resolution=(448, 448)),
}
