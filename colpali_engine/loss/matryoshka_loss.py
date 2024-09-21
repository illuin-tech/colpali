from typing import List, Optional

import torch
import torch.nn as nn


class MatryoshkaCELoss(nn.Module):
    """
    Loss function for Matryoshka Representation Learning (MRL).

    This loss computes the cross-entropy loss for embeddings at multiple dimensionalities and aggregates them,
    optionally weighting each loss by a specified relative importance.

    Args:
    - relative_importance (Optional[List[float]]):
        A list of weights for each embedding dimensionality's loss.
        If None, all dimensions are equally weighted.
    - **kwargs:
        Additional keyword arguments to pass to `nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        relative_importance: Optional[List[float]] = None,
        **kwargs,
    ):
        super(MatryoshkaCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        self.relative_importance = relative_importance

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute the Matryoshka Loss.

        Args:
        - Input: Tensor of shape `(batch_size, num_classes)`
        - Target: Tensor of shape `(batch_size,)` containing class indices.

        Returns:
        - torch.Tensor: The aggregated loss value.
        """

        # Sanity checks
        assert input.dim() == 2, f"Expected input to have 2 dimensions, got {input.dim()} dimensions."

        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze(1)
        assert target.dim() == 1, f"Expected target to have 1 dimension, got {target.dim()} dimensions."

        # Number of embedding dimensionalities
        num_dimensions = input.shape[0]

        # Determine weights for each dimensionality
        if self.relative_importance is None:
            weights = [1.0] * num_dimensions
        else:
            if len(self.relative_importance) != num_dimensions:
                raise ValueError(
                    f"Length of relative_importance ({len(self.relative_importance)}) "
                    f"must match number of dimensions ({num_dimensions})."
                )
            weights = self.relative_importance

        # Compute loss for each dimensionality
        total_loss = torch.tensor(0.0, dtype=torch.float)
        for i in range(num_dimensions):
            loss = self.criterion.forward(input=input[i], target=target)
            weighted_loss = weights[i] * loss
            total_loss += weighted_loss

        return total_loss
