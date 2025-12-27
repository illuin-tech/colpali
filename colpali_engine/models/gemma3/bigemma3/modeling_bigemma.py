from typing import ClassVar, Literal

import torch
from transformers.models.gemma3 import Gemma3Config, Gemma3Model


class BiGemma3(Gemma3Model):  # noqa: N801
    """
    BiGemma3 is an implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    Representations are pooled to obtain a single vector representation. Based on the Gemma3 backbone.

    Supports Matryoshka embeddings with dimensions: 768, 1536, or 2560 (full).
    The embedding dimension can be specified at inference time via the forward() method.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Gemma3Config):
        super().__init__(config=config)
        self.padding_side = "left"
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Remove embedding_dim if passed (backward compatibility)
        kwargs.pop("embedding_dim", None)

        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping

        # Load the model without embedding_dim parameter
        model = super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)
        return model

    def forward(
        self,
        pooling_strategy: Literal["cls", "last", "mean"] = "last",
        embedding_dim: int = 2560,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for BiGemma3 model.

        Args:
            pooling_strategy: The strategy to use for pooling the hidden states.
            embedding_dim: Matryoshka dimension (768, 1536, or 2560). Default: 2560 (full).
            *args: Variable length argument list.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Dense embeddings (batch_size, embedding_dim).
        """
        # Validate embedding_dim
        if embedding_dim not in [768, 1536, 2560]:
            raise ValueError(f"embedding_dim must be one of [768, 1536, 2560], got {embedding_dim}")

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)

        outputs = super().forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
        last_hidden_states = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        # Get CLS token embedding, last token, or mean pool over sequence
        if pooling_strategy == "cls":
            # Use CLS token (first token) embedding
            pooled_output = last_hidden_states[:, 0]  # (batch_size, hidden_size)
        elif pooling_strategy == "last":
            # use last token since we are left padding
            pooled_output = last_hidden_states[:, -1]  # (batch_size, hidden_size)
        elif pooling_strategy == "mean":
            # Mean pooling over sequence length
            if "attention_mask" in kwargs:
                mask = kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, 1)
                pooled_output = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, hidden_size)
            else:
                pooled_output = last_hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        else:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

        # Matryoshka: slice to the desired embedding dimension
        pooled_output = pooled_output[:, :embedding_dim]  # (batch_size, embedding_dim)

        # L2 normalization
        pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=-1)
        return pooled_output
