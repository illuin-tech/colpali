from typing import ClassVar, List

import torch
import torch.nn as nn


class MultiVectorPooler(nn.Module):
    supported_pooling_strategies: ClassVar[List[str]] = ["mean", "sum", "max"]

    def __init__(self, pooling_strategy: str = "mean"):
        """
        Initialize the MultiVectorPooler with a specified pooling strategy.

        Args:
        - pooling_strategy (SupportedPoolingType): The type of pooling to apply.
        """
        super().__init__()

        if pooling_strategy not in self.supported_pooling_strategies:
            raise ValueError(
                f"Unsupported pooling type: {pooling_strategy}. Use one of {self.supported_pooling_strategies}."
            )
        self.pooling_strategy = pooling_strategy

    def forward(self, input_tensor) -> torch.Tensor:
        """
        Apply the pooling operation on the input tensor.

        Args:
        - input_tensor (torch.Tensor): A 3D tensor with shape (batch_size, num_tokens, dim).

        Returns:
        - torch.Tensor: A 2D tensor with shape (batch_size, dim) after pooling.
        """
        if self.pooling_strategy == "mean":
            pooled_tensor = torch.mean(input_tensor, dim=1)
        elif self.pooling_strategy == "sum":
            pooled_tensor = torch.sum(input_tensor, dim=1)
        elif self.pooling_strategy == "max":
            pooled_tensor, _ = torch.max(input_tensor, dim=1)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}.")

        return pooled_tensor
