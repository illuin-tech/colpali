from enum import Enum

import torch
import torch.nn as nn


class MultiVectorPoolingStrategy(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"


class MultiVectorPooler(nn.Module):
    def __init__(self, pooling_strategy: str = MultiVectorPoolingStrategy.MEAN):
        """
        Initialize the MultiVectorPooler with a specified pooling strategy.

        Args:
        - pooling_strategy (SupportedPoolingType): The type of pooling to apply.
        """
        super(MultiVectorPooler, self).__init__()
        if not isinstance(pooling_strategy, MultiVectorPoolingStrategy):
            raise ValueError(
                f"Unsupported pooling type: {pooling_strategy}. Use one of {list(MultiVectorPoolingStrategy)}."
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
        if self.pooling_strategy == MultiVectorPoolingStrategy.MEAN:
            pooled_tensor = torch.mean(input_tensor, dim=1)
        elif self.pooling_strategy == MultiVectorPoolingStrategy.SUM:
            pooled_tensor = torch.sum(input_tensor, dim=1)
        elif self.pooling_strategy == MultiVectorPoolingStrategy.MAX:
            pooled_tensor, _ = torch.max(input_tensor, dim=1)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}.")

        return pooled_tensor
