from dataclasses import dataclass
from typing import Dict


@dataclass(kw_only=True)
class ViTConfig:
    patch_size: int
    resolution: int

    @property
    def n_patch_per_dim(self) -> int:
        if self.resolution % self.patch_size != 0:
            raise ValueError(f"Resolution {self.resolution} is not divisible by patch size {self.patch_size}")
        return self.resolution // self.patch_size


VIT_CONFIG: Dict[str, ViTConfig] = {
    "vidore/colpali": ViTConfig(patch_size=14, resolution=448),
    "vidore/colpali-v1.1": ViTConfig(patch_size=14, resolution=448),
    "vidore/colpali-v1.2": ViTConfig(patch_size=14, resolution=448),
    "vidore/colpali-v1.2-merged": ViTConfig(patch_size=14, resolution=448),
}
