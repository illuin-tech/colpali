from typing import Literal

import torch
from transformers import Idefics3Config, Idefics3Model, Idefics3PreTrainedModel


class BiIdefics3(Idefics3PreTrainedModel):
    """
    Initializes the BiIdefics3 model.

    Args:
        config : The model configuration.
    """

    def __init__(self, config: Idefics3Config):
        super(BiIdefics3, self).__init__(config=config)
        self.model: Idefics3Model = Idefics3Model(config)
        self.padding_side = "left"
        self.post_init()

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
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

        # Get CLS token embedding, last token, or mean pool over sequence
        if pooling_strategy == "cls":
            # Use CLS token (first token) embedding
            pooled_output = last_hidden_states[:, 0]  # (batch_size, hidden_size)
        elif pooling_strategy == "last":
            # use last token since we are left padding
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
