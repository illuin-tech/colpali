from typing import cast

import torch
from torch import nn
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel

from colpali_engine.models.colpali_2.colpali_2_config import ColPali2Config
from colpali_engine.models.colpali_2.colpali_2_modeling_outputs import ColPali2ModelOutput
from colpali_engine.models.colpali_2.colpali_2_utils import MultiVectorPooler


class ColPali2(PaliGemmaPreTrainedModel):
    def __init__(self, config: ColPali2Config):
        super(ColPali2, self).__init__(config=config)
        self.config = cast(ColPali2Config, self.config)
        self.model = PaliGemmaForConditionalGeneration(self.config.vlm_config)

        self.single_vector_projector = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.multi_vector_pooler = MultiVectorPooler(pooling_strategy=self.config.single_vector_pool_strategy)
        self.multi_vector_projector = nn.Linear(self.model.config.text_config.hidden_size, self.dim)

        self.main_input_name = "doc_input_ids"

    @property
    def single_vector_projector_dim(self) -> int:
        return self.config.single_vector_projector_dim

    @property
    def multi_vector_projector_dim(self) -> int:
        return self.config.multi_vector_projector_dim

    def forward(self, *args, **kwargs) -> ColPali2ModelOutput:
        """
        Forward pass through ColPali. Returns both single-vector and multi-vector embeddings.

        NOTE: Both the text and image processors should prepend the <CLS> token to the input_ids tensor
        before passing it to the model.

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - ColPaliModelOutput:
            - single_vector (torch.Tensor): Single-vector embeddings of shape (batch_size, dim).
            - multi_vector (torch.Tensor): Multi-vector embeddings of shape (batch_size, num_tokens, dim).
        """

        # Forward pass through the VLM
        vlm_outputs = self.model(*args, output_hidden_states=True, **kwargs)
        vlm_last_hidden_states = vlm_outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)

        # Head 1: Single-vector embedding
        pooled_output = self.multi_vector_pooler(vlm_last_hidden_states)  # (batch_size, hidden_size)
        single_vec_emb = self.single_vector_projector(pooled_output)  # (batch_size, hidden_size)
        single_vec_emb = torch.nn.functional.normalize(single_vec_emb, dim=-1)

        # Head 2: Multi-vector embedding
        multi_vec_emb = self.multi_vector_projector(
            vlm_last_hidden_states[:, 1:, :]
        )  # (batch_size, sequence_length, hidden_size)
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        multi_vec_emb = multi_vec_emb * kwargs["attention_mask"].unsqueeze(-1)

        return ColPali2ModelOutput(single_vec_emb=single_vec_emb, multi_vec_emb=multi_vec_emb)

    def forward_single_vector(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through ColPali. Returns only the single-vector embeddings.
        """
        vlm_outputs = self.model(*args, output_hidden_states=True, **kwargs)
        pooled_output = self.multi_vector_pooler(vlm_outputs.hidden_states[-1])  # (batch_size, hidden_size)
        single_vec_emb = self.single_vector_projector(pooled_output)  # (batch_size, hidden_size)
        single_vec_emb = torch.nn.functional.normalize(single_vec_emb, dim=-1)

        return single_vec_emb

    def forward_multi_vector(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through ColPali. Returns only the multi-vector embeddings.
        """
        vlm_outputs = self.model(*args, output_hidden_states=True, **kwargs)
        multi_vec_emb = self.multi_vector_projector(
            vlm_outputs.hidden_states[-1]
        )  # (batch_size, sequence_length, hidden_size)
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        multi_vec_emb = multi_vec_emb * kwargs["attention_mask"].unsqueeze(-1)

        return multi_vec_emb
