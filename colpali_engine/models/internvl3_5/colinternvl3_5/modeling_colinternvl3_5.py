from typing import ClassVar

import torch
from torch import nn
from transformers.models.internvl import InternVLConfig, InternVLModel




class ColInternVL3_5(InternVLModel):
    """
    ColInternVL3_5 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Args:
        config (InternVLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: InternVLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        # breakpoint()
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        )  # (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
