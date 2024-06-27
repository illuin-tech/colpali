import os
from typing import Optional

import torch
from transformers import SiglipModel


class SigLIP(SiglipModel):
    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        return self.forward_branch(*args, **kwargs)

    def forward_branch(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is not None:
            # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.

            outputs = self.vision_model(
                pixel_values=pixel_values.to(dtype=self.dtype),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        else:
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        embeds = outputs[1]

        # normalized features
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        return embeds


class ColSigLIP(SiglipModel):
    def __init__(self, config):
        super(ColSigLIP, self).__init__(config=config)
        self.dim = 128
        self.custom_vision_proj = torch.nn.Linear(self.config.vision_config.hidden_size, self.dim)
        self.custom_text_proj = torch.nn.Linear(self.config.text_config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        return self.forward_branch(*args, **kwargs)

    def forward_branch(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is not None:
            # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.

            outputs = self.vision_model(
                pixel_values=pixel_values.to(dtype=self.dtype),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

            last_hidden_states = outputs.last_hidden_state

            proj = self.custom_vision_proj(last_hidden_states)
            # normalize l2 norm
            proj = proj / proj.norm(dim=-1, keepdim=True)

        else:
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_states = outputs.last_hidden_state

            proj = self.custom_text_proj(last_hidden_states)
            # normalize l2 norm
            proj = proj / proj.norm(dim=-1, keepdim=True)
            proj = proj * attention_mask.unsqueeze(-1)

        # normalized features
        return proj
