from typing import List, Optional

import torch


class MatryoshkaCELoss(torch.nn.Module):
    """
    Loss function for Matryoshka Representation Learning.

    Adapted from https://github.com/RAIVNLab/MRL/blob/7ccb42df6be05f3d21d0648aa03099bba46386bf/MRL.py#L11
    """

    def __init__(self, relative_importance: Optional[List[float]] = None, **kwargs):
        super(MatryoshkaCELoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(**kwargs)
        self.relative_importance = relative_importance

    def forward(self, output, target) -> torch.Tensor:
        # Calculate losses for each output and stack them. This is still O(N)
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])

        # Set relative_importance to 1 if not specified
        rel_importance = (
            torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
        )

        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()
