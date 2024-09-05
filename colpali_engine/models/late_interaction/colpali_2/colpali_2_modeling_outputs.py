from dataclasses import dataclass

import torch


@dataclass
class ColPali2ModelOutput:
    single_vec_emb: torch.Tensor
    multi_vec_emb: torch.Tensor
