from transformers import SiglipModel


class BiSiglip(SiglipModel):
    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """

        output_attentions = kwargs.pop("output_attentions", None)
        output_hidden_states = kwargs.pop("output_hidden_states", None)
        return_dict = kwargs.pop("return_dict", None)
        interpolate_pos_encoding = kwargs.pop("interpolate_pos_encoding", None)

        if "pixel_values" in kwargs:
            # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
            pixel_values = kwargs.pop("pixel_values")

            embeds = self.vision_model(
                pixel_values=pixel_values.to(dtype=self.dtype),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                interpolate_pos_encoding=interpolate_pos_encoding,
            ).pooler_output

        else:
            embeds = self.text_model(
                input_ids=kwargs.pop("input_ids", None),
                attention_mask=kwargs.pop("attention_mask", None),
                position_ids=kwargs.pop("position_ids", None),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).pooler_output

        # normalized features
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        return embeds
