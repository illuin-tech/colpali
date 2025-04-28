from typing import Literal, ClassVar

import torch
from torch import nn
from transformers import Idefics3Model, Idefics3PreTrainedModel


class BiIdefics3(Idefics3PreTrainedModel):
    """
    Initializes the BiIdefics3 model.

    Args:
        config : The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
        except those of the image at inference.
        Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config, mask_non_image_embeddings: bool = False):
        super(BiIdefics3, self).__init__(config=config)
        self.model: Idefics3Model = Idefics3Model(config)
        self.mask_non_image_embeddings = mask_non_image_embeddings

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            pooling_strategy: The strategy to use for pooling the hidden states.
            *args: Variable length argument list.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Dense embeddings (batch_size, hidden_size).
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
            if "pixel_values" in kwargs and self.mask_non_image_embeddings:
                # Pools only the image embeddings
                mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
                pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
            else:
                # Mean pooling over sequence length
                mask = kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, 1)
                pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        # L2 normalization
        pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)
        return pooled_output
