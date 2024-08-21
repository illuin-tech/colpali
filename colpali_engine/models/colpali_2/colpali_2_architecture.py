import torch
from torch import nn
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel

from colpali_engine.models.colpali_2.colpali_2_modeling_outputs import ColPali2ModelOutput


class ColPali2(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(ColPali2, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        self.dim = 128
        self.single_vector_projector = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.multi_vector_projector = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"

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
        cls_last_hidden_state = vlm_last_hidden_states[:, 0, :]  # (batch_size, hidden_size)
        single_vec_emb = self.single_vector_projector(cls_last_hidden_state)  # (batch_size, hidden_size)
        single_vec_emb = torch.nn.functional.normalize(single_vec_emb, dim=-1)

        # Head 2: Multi-vector embedding
        multi_vec_emb = self.multi_vector_projector(
            vlm_last_hidden_states[:, 1:, :]
        )  # (batch_size, sequence_length, hidden_size)
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        multi_vec_emb = multi_vec_emb * kwargs["attention_mask"].unsqueeze(-1)

        return ColPali2ModelOutput(single_vector=single_vec_emb, multi_vector=multi_vec_emb)

    def forward_single_vector(self, *args, **kwargs):
        """
        Forward pass through ColPali. Returns only the single-vector embeddings.
        """
        vlm_outputs = self.model(*args, output_hidden_states=True, **kwargs)
        cls_last_hidden_state = vlm_outputs.hidden_states[-1][:, 0, :]  # (batch_size, hidden_size)
        single_vec_emb = self.single_vector_projector(cls_last_hidden_state)  # (batch_size, hidden_size)
        single_vec_emb = torch.nn.functional.normalize(single_vec_emb, dim=-1)

        return single_vec_emb

    def forward_multi_vector(self, *args, **kwargs):
        """
        Forward pass through ColPali. Returns only the multi-vector embeddings.
        """
        vlm_outputs = self.model(*args, output_hidden_states=True, **kwargs)
        multi_vec_emb = self.multi_vector_projector(
            vlm_outputs.hidden_stages[-1][:, 1:, :]
        )  # (batch_size, sequence_length, hidden_size)
        multi_vec_emb = torch.nn.functional.normalize(multi_vec_emb, dim=-1)
        multi_vec_emb = multi_vec_emb * kwargs["attention_mask"].unsqueeze(-1)

        return multi_vec_emb
