from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch
from torch import nn
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaCausalLMOutputWithPast,
    PaliGemmaForConditionalGeneration,
)

from colpali_engine.models.paligemma.colpali_duo.configuration_colpali_duo import (
    ColPaliDuoConfig,
    to_vlm_backbone_config,
)


@dataclass
class ColPaliDuoLossOutputs:
    """
    Base class for ColPaliDuo embeddings output.
    """

    single_vector_loss: torch.Tensor
    multi_vector_loss: torch.Tensor
    loss: torch.Tensor
    distillation_loss: Optional[torch.Tensor] = None


@dataclass
class ColPaliDuoModelOutput(PaliGemmaCausalLMOutputWithPast):
    """
    Base class for the ColPaliDuo outputs.

    Args:
        vlm_last_hidden_states (torch.Tensor, optional): Last hidden states of the VLM.
        single_vec_emb (torch.Tensor, optional): Single-vector embeddings.
        multi_vec_emb (torch.Tensor, optional): Multi-vector embeddings.
        loss_duo (torch.FloatTensor, optional): Loss tensor.
    """

    vlm_last_hidden_states: Optional[torch.Tensor] = None
    single_vec_emb: Optional[torch.Tensor] = None
    multi_vec_emb: Optional[torch.Tensor] = None
    loss_duo: Optional[ColPaliDuoLossOutputs] = None


class ColPaliDuo(PaliGemmaForConditionalGeneration):
    """
    ColPali model with two heads that can generate both single-vector and multi-vector embeddings.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-relat ed

    def __init__(self, config: ColPaliDuoConfig):
        vlm_backbone_config = to_vlm_backbone_config(config)
        super().__init__(vlm_backbone_config)

        # Add the additional attributes
        self.config.single_vector_projector_dim = config.single_vector_projector_dim
        self.config.single_vector_pool_strategy = config.single_vector_pool_strategy
        self.config.multi_vector_projector_dim = config.multi_vector_projector_dim

        self.single_vector_projector = nn.Linear(
            in_features=self.config.text_config.hidden_size,
            out_features=self.config.single_vector_projector_dim,
        )

        # self.multi_vector_pooler = MultiVectorPooler(pooling_strategy=self.config.single_vector_pool_strategy)
        self.multi_vector_projector = nn.Linear(
            in_features=self.config.text_config.hidden_size,
            out_features=self.config.multi_vector_projector_dim,
        )

    @property
    def single_vector_projector_dim(self) -> int:
        return self.config.single_vector_projector_dim

    @property
    def multi_vector_projector_dim(self) -> int:
        return self.config.multi_vector_projector_dim

    @staticmethod
    def _delete_redundant_forward_kwargs(kwargs: Dict[str, Any]) -> None:
        """
        Delete redundant kwargs before passing them to the forward method. In-place operation.
        """
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("output_hidden_states", None)

    def get_vlm_last_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the VLM. Returns the last hidden states.

        Output shape: (batch_size, seq_length, hidden_size).
        """
        vlm_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        vlm_last_hidden_states = vlm_outputs.hidden_states[-1]  # (batch_size, seq_length, hidden_size)

        return vlm_last_hidden_states

    def project_to_single_vector_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project the hidden states to single-vector embeddings.

        Returns:
            torch.Tensor: Single-vector embeddings of shape (batch_size, hidden_size).
        """
        # pooled_output = self.multi_vector_pooler(hidden_states)  # (batch_size, hidden_size)
        pooled_output = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(
            attention_mask, dim=1, keepdim=True
        )
        single_vec_emb = self.single_vector_projector(pooled_output)  # (batch_size, hidden_size)
        single_vec_emb = torch.nn.functional.normalize(single_vec_emb, dim=-1)  # (batch_size, hidden_size)

        return single_vec_emb

    def project_to_multi_vector_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project the hidden states to multi-vector embeddings.

        Returns:
            torch.Tensor: Multi-vector embeddings of shape (batch_size, num_tokens, hidden_size).
        """
        multi_vec_emb = self.multi_vector_projector(hidden_states)  # (batch_size, seq_length, hidden_size)
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)  # (batch_size, seq_length, hidden_size)
        multi_vec_emb = multi_vec_emb * attention_mask.unsqueeze(-1)  # (batch_size, seq_length, hidden_size)

        return multi_vec_emb

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        output_vlm_last_hidden_states: bool = False,
        *args,
        **kwargs,
    ) -> ColPaliDuoModelOutput:
        """
        Forward pass through ColPaliDuo. Returns both single-vector and multi-vector embeddings.

        Args:
            input_ids (torch.LongTensor): The input tokens tensor.
            attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
            ColPaliModelOutput:
                single_vector (torch.Tensor): Single-vector embeddings of shape (batch_size, dim).
                multi_vector (torch.Tensor): Multi-vector embeddings of shape (batch_size, num_tokens, dim).
        """
        # Delete redundant kwargs
        self._delete_redundant_forward_kwargs(kwargs)

        # Forward pass through the VLM
        hidden_states = self.get_vlm_last_hidden_states(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )  # (batch_size, seq_length, hidden_size)

        # Compute the embeddings
        single_vec_emb = self.project_to_single_vector_embeddings(hidden_states, attention_mask)
        multi_vec_emb = self.project_to_multi_vector_embeddings(hidden_states, attention_mask)

        return ColPaliDuoModelOutput(
            vlm_last_hidden_states=hidden_states if output_vlm_last_hidden_states else None,
            single_vec_emb=single_vec_emb,
            multi_vec_emb=multi_vec_emb,
        )

    def forward_single_vector(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        output_vlm_last_hidden_states: bool = False,
        **kwargs,
    ) -> ColPaliDuoModelOutput:
        """
        Forward pass through ColPaliDuo. Returns only the single-vector embeddings.
        """
        # Delete redundant kwargs
        self._delete_redundant_forward_kwargs(kwargs)

        # Forward pass through the VLM
        hidden_states = self.get_vlm_last_hidden_states(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Compute the embeddings
        single_vec_emb = self.project_to_single_vector_embeddings(hidden_states)

        return ColPaliDuoModelOutput(
            vlm_last_hidden_states=hidden_states if output_vlm_last_hidden_states else None,
            single_vec_emb=single_vec_emb,
        )

    def forward_multi_vector(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        output_vlm_last_hidden_states: bool = False,
        **kwargs,
    ) -> ColPaliDuoModelOutput:
        """
        Forward pass through ColPaliDuo. Returns only the multi-vector embeddings.
        """
        # Delete redundant kwargs
        self._delete_redundant_forward_kwargs(kwargs)

        # Forward pass through the VLM
        hidden_states = self.get_vlm_last_hidden_states(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Compute the embeddings
        multi_vec_emb = self.project_to_multi_vector_embeddings(hidden_states, attention_mask)

        return ColPaliDuoModelOutput(
            vlm_last_hidden_states=hidden_states if output_vlm_last_hidden_states else None,
            multi_vec_emb=multi_vec_emb,
        )
