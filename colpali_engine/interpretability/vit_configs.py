from dataclasses import dataclass
from typing import Dict


@dataclass
class ViTConfig:
    patch_size: int
    resolution: int

    @property
    def n_patch_per_dim(self) -> int:
        if self.resolution % self.patch_size != 0:
            raise ValueError(f"Resolution {self.resolution} is not divisible by patch size {self.patch_size}")
        return self.resolution // self.patch_size


VIT_CONFIG: Dict[str, ViTConfig] = {
    "google/siglip-so400m-patch14-384": ViTConfig(patch_size=14, resolution=384),
    "timm/ViT-SO400M-14-SigLIP-384": ViTConfig(patch_size=14, resolution=384),
    "google/paligemma-3b-mix-448": ViTConfig(
        patch_size=14, resolution=448
    ),  # based on "timm/ViT-SO400M-14-SigLIP-384" with increased resolution
}
