from typing import Optional

import torch
from torch import nn
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel


class BiPaliProj(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(BiPaliProj, self).__init__(config=config)
        model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model: PaliGemmaForConditionalGeneration = model
        self.main_input_name = "doc_input_ids"
        self.dim = 1024
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings with Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings with Llava->PaliGemma
    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings with Llava->PaliGemma
    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder with Llava->PaliGemma
    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder with Llava->PaliGemma
    def get_decoder(self):
        return self.model.language_model.get_decoder()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights with Llava->PaliGemma
    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        # delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)

        # pooling -mean on attention mask==1
        proj = torch.sum(last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1) / torch.sum(
            kwargs["attention_mask"], dim=1, keepdim=True
        )
        proj = self.custom_text_proj(proj)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj


class BiPali(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(BiPali, self).__init__(config=config)
        model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model: PaliGemmaForConditionalGeneration = model
        self.main_input_name = "doc_input_ids"
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings with Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings with Llava->PaliGemma
    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings with Llava->PaliGemma
    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder with Llava->PaliGemma
    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder with Llava->PaliGemma
    def get_decoder(self):
        return self.model.language_model.get_decoder()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights with Llava->PaliGemma
    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        # delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        # pooling -mean on attention mask==1
        proj = torch.sum(last_hidden_states * kwargs["attention_mask"].unsqueeze(-1), dim=1) / torch.sum(
            kwargs["attention_mask"], dim=1, keepdim=True
        )
        proj = proj / proj.norm(dim=-1, keepdim=True)
        return proj


class ColPali(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(ColPali, self).__init__(config=config)
        model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.main_input_name = "doc_input_ids"
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings with Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings with Llava->PaliGemma
    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings with Llava->PaliGemma
    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder with Llava->PaliGemma
    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder with Llava->PaliGemma
    def get_decoder(self):
        return self.model.language_model.get_decoder()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights with Llava->PaliGemma
    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        # delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj


class ColNewSiglip(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(ColNewSiglip, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = PaliGemmaForConditionalGeneration(config)
        self.dim = 128
        self.custom_image_proj = nn.Linear(self.model.config.vision_config.projection_dim, self.dim)
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
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
            proj = self.custom_image_proj(image_features)
            # print(f"Doc proj: {proj.shape}")
            proj = proj / proj.norm(dim=-1, keepdim=True)
        else:
            # delete output_hidden_states from kwargs
            kwargs.pop("output_hidden_states", None)
            outputs = self.model(*args, output_hidden_states=True, **kwargs)
            last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
            # print(f"Query: {last_hidden_states.shape}")
            proj = self.custom_text_proj(last_hidden_states)
            # print(f"Query proj: {proj.shape}")
            # normalize l2 norm
            proj = proj / proj.norm(dim=-1, keepdim=True)
            proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj

    def vision_model_output(self, input_ids: torch.LongTensor = None, pixel_values: torch.FloatTensor = None, **kwargs):

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.model.vision_tower(pixel_values.to(inputs_embeds.dtype))
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.model.multi_modal_projector(selected_image_feature)

            return image_features

        raise ValueError("pixel_values is None or input_ids.shape[1] == 1")


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

    def vision_model_output(self, input_ids: torch.LongTensor = None, pixel_values: torch.FloatTensor = None, **kwargs):

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.model.vision_tower(pixel_values.to(inputs_embeds.dtype))
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.model.multi_modal_projector(selected_image_feature)

            return image_features

        raise ValueError("pixel_values is None or input_ids.shape[1] == 1")
