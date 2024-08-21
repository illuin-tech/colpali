from dataclasses import dataclass

import torch


@dataclass
class ColPali2ModelOutput:
    single_vector: torch.Tensor
    multi_vector: torch.Tensor
