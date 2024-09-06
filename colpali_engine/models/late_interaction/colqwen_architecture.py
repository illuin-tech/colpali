from typing import List, Optional

import torch
from torch import nn
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLPreTrainedModel,
    Qwen2VLModel,
)


class ColQwen(Qwen2VLForConditionalGeneration):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    def __init__(self, config: Qwen2VLConfig):
        super(ColQwen, self).__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        output_hidden_states = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.custom_text_proj(hidden_states)
        proj = logits.float()

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        proj = proj * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

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
