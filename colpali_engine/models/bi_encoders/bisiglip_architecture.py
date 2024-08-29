from typing import Optional

import torch
from transformers import SiglipModel
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel


class BiSigLIP(SiglipModel):
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


class BiNewSiglip(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(BiNewSiglip, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
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
        # outputs = self.model(*args, output_hidden_states=True, **kwargs)
        if "pixel_values" in kwargs:
            image_features = self.vision_model_output(*args, **kwargs)
            # print(f"Doc: {image_features.shape}")
            # pool image features
            proj = torch.mean(image_features, dim=1)
            # print(f"Doc proj: {proj.shape}")
            norm = proj.norm(dim=-1, keepdim=True)
            proj = proj / norm
        else:
            # delete output_hidden_states from kwargs
            kwargs.pop("output_hidden_states", None)
            outputs = self.model(*args, output_hidden_states=True, **kwargs)
            last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
            # pooling -mean on attention mask==1

            proj = torch.sum(last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1) / torch.sum(
                kwargs["attention_mask"], dim=1, keepdim=True
            )
            # print(f"Query proj: {proj.shape}")
            norm = proj.norm(dim=-1, keepdim=True)
            proj = proj / norm
        return proj

    def vision_model_output(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.model.vision_tower(pixel_values.to(inputs_embeds.dtype))
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.model.multi_modal_projector(selected_image_feature)

            return image_features

        raise ValueError("pixel_values is None or input_ids.shape[1] == 1")
