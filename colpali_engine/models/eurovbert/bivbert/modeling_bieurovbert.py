import os
import torch

from typing import Literal, Union

from colpali_engine.models.eurovbert.modeling_vbert import VBertModel, VBertPreTrainedModel
from colpali_engine.models.eurovbert.configuration_vbert import VBertConfig


class BiEuroVBert(VBertPreTrainedModel):
    """
    Initializes the BiIdefics3 model.

    Args:
        config : The model configuration.
    """
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, pooling_strategy = "mean", **kwargs):
        super().__init__(config=config)
        self.model = VBertModel(config, **kwargs)
        self.pooling_strategy = pooling_strategy
        self.post_init()

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through model and pooling.

        Args:
        - pooling_strategy (str): The pooling strategy to use. Options are "cls", "last", or "mean".
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, dim)
        """
        outputs = self.model(*args, **kwargs)
        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)

        pooling_strategy = pooling_strategy or self.pooling_strategy

        # Get CLS token embedding, last token, or mean pool over sequence
        if pooling_strategy == "cls":
            # Use CLS token (first token) embedding
            pooled_output = last_hidden_states[:, 0]  # (batch_size, hidden_size)
        elif pooling_strategy == "last":
            # Use last token
            pooled_output = last_hidden_states[:, -1]  # (batch_size, hidden_size)
        elif pooling_strategy == "mean":
            # Mean pooling over sequence length
            mask = kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, 1)
            pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        # L2 normalization
        pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)
        return pooled_output
