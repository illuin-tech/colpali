from typing import Optional

import torch
from torch import nn
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLPreTrainedModel, Qwen2VLModel


class ColQwen(Qwen2VLPreTrainedModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    def __init__(self, config: Qwen2VLConfig):
        super(ColQwen, self).__init__(config=config)
        model = Qwen2VLForConditionalGeneration(config)
        if model.model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.model.{k}" for k in model.model._tied_weights_keys]
        self.model = model
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return proj

    # def get_input_embeddings(self):
    #     return self.model.model.get_input_embeddings()

    # def set_input_embeddings(self, value):
    #     self.model.model.set_input_embeddings(value)

    # def get_output_embeddings(self):
    #     return self.model.model.get_output_embeddings()

    # def set_output_embeddings(self, new_embeddings):
    #     self.model.model.set_output_embeddings(new_embeddings)

    # def set_decoder(self, decoder):
    #     self.model.model.set_decoder(decoder)

    # def get_decoder(self):
    #     return self.model.omde.get_decoder()

    # def tie_weights(self):
    #     return self.model.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings

        return model_embeds
