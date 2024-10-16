from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Type

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.paligemma import PaliGemmaForConditionalGeneration
from transformers.utils import ModelOutput

from colpali_engine.compression.pooling.multi_vector_pooler import MultiVectorPooler
from colpali_engine.models.paligemma.colpali_duo.configuration_colpali_duo import ColPaliDuoConfig


@dataclass(kw_only=True)
class ColPaliDuoLossOutputs:
    """
    Base class for ColPaliDuo embeddings output.
    """

    single_vector_loss: torch.FloatTensor
    multi_vector_loss: torch.FloatTensor
    distillation_loss: Optional[torch.FloatTensor] = None
    total_loss: torch.FloatTensor


@dataclass(kw_only=True)
class ColPaliDuoModelOutput(ModelOutput):
    """
    Base class for the ColPaliDuo outputs.

    Args:
    - loss (torch.FloatTensor, optional): Loss tensor.
    - vlm_last_hidden_states (torch.Tensor, optional): Last hidden states of the VLM.
    - single_vec_emb (torch.Tensor, optional): Single-vector embeddings.
    - multi_vec_emb (torch.Tensor, optional): Multi-vector embeddings.
    """

    vlm_last_hidden_states: Optional[torch.Tensor] = None
    single_vec_emb: Optional[torch.Tensor] = None
    multi_vec_emb: Optional[torch.Tensor] = None
    loss: Optional[ColPaliDuoLossOutputs] = None


class ColVisionDuo(PreTrainedModel, ABC):
    """
    Base class for the ColVisionDuo architecture.

    If your VLM backbone has a non-standard forward method, you can override the ColVisionDuo methods accordingly.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    @property
    @abstractmethod
    def vlm_class(self) -> Type[PreTrainedModel]:
        pass

    def __init__(self, config: ColPaliDuoConfig):
        if not hasattr(self, "vlm_class"):
            raise ValueError("The `vlm_class` attribute must be defined in the child ColVisionDuo class.")

        super().__init__(config=config)

        self.vlm_backbone = self.vlm_class(self.config.vlm_backbone_config)

        self.single_vector_projector = nn.Linear(
            in_features=self.vlm_backbone.config.text_config.hidden_size,
            out_features=self.config.single_vector_projector_dim,
        )

        self.multi_vector_pooler = MultiVectorPooler(pooling_strategy=self.config.single_vector_pool_strategy)
        self.multi_vector_projector = nn.Linear(
            in_features=self.vlm_backbone.config.text_config.hidden_size,
            out_features=self.config.multi_vector_projector_dim,
        )

    @property
    def single_vector_projector_dim(self) -> int:
        return self.config.single_vector_projector_dim

    @property
    def multi_vector_projector_dim(self) -> int:
        return self.config.multi_vector_projector_dim

    @staticmethod
    def _prepare_forward_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = kwargs.copy()
        kwargs.pop("input_ids", None)
        kwargs.pop("attention_mask", None)
        kwargs.pop("output_hidden_states", None)
        return kwargs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_vlm_last_hidden_states: bool = False,
        **kwargs,
    ) -> ColPaliDuoModelOutput:
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
        # Delete redundant kwargs
        kwargs = self._prepare_forward_kwargs(kwargs)

        # Forward pass through the VLM
        vlm_outputs = self.vlm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
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
        multi_vec_emb = multi_vec_emb * attention_mask.unsqueeze(-1)

        return ColPaliDuoModelOutput(
            vlm_last_hidden_states=vlm_last_hidden_states if output_vlm_last_hidden_states else None,
            single_vec_emb=single_vec_emb,
            multi_vec_emb=multi_vec_emb,
        )

    def forward_single_vector(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_vlm_last_hidden_states: bool = False,
        **kwargs,
    ) -> ColPaliDuoModelOutput:
        """
        Forward pass through ColPali. Returns only the single-vector embeddings.
        """
        # Delete redundant kwargs
        kwargs = self._prepare_forward_kwargs(kwargs)

        # Forward pass through the VLM
        vlm_outputs = self.vlm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        vlm_last_hidden_states = vlm_outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)

        # Forward pass through the single-vector head
        pooled_output = self.multi_vector_pooler(vlm_last_hidden_states)  # (batch_size, hidden_size)
        single_vec_emb = self.single_vector_projector(pooled_output)  # (batch_size, hidden_size)
        single_vec_emb = torch.nn.functional.normalize(single_vec_emb, dim=-1)

        return ColPaliDuoModelOutput(
            vlm_last_hidden_states=vlm_last_hidden_states if output_vlm_last_hidden_states else None,
            single_vec_emb=single_vec_emb,
        )

    def forward_multi_vector(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_vlm_last_hidden_states: bool = False,
        **kwargs,
    ) -> ColPaliDuoModelOutput:
        """
        Forward pass through ColPali. Returns only the multi-vector embeddings.
        """
        # Delete redundant kwargs
        kwargs = self._prepare_forward_kwargs(kwargs)

        # Forward pass through the VLM
        vlm_outputs = self.vlm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        vlm_last_hidden_states = vlm_outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)

        # Forward pass through the multi-vector head
        multi_vec_emb = self.multi_vector_projector(
            vlm_last_hidden_states
        )  # (batch_size, sequence_length, hidden_size)
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        multi_vec_emb = multi_vec_emb * attention_mask.unsqueeze(-1)

        return ColPaliDuoModelOutput(
            vlm_last_hidden_states=vlm_last_hidden_states if output_vlm_last_hidden_states else None,
            multi_vec_emb=multi_vec_emb,
        )


class ColPaliDuo(ColVisionDuo):
    """
    ColVisionDuo with the PaliGemma model as the VLM backbone.
    """

    @property
    def vlm_class(self):
        return PaliGemmaForConditionalGeneration

    def __init__(self, config: ColPaliDuoConfig):
        super().__init__(config=config)
