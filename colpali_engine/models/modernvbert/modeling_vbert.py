from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, PreTrainedModel, logging
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput

from .configuration_vbert import VBertConfig

logger = logging.get_logger(__name__)

torch.set_float32_matmul_precision("high")


class DecoupledEmbedding(nn.Embedding):
    """
    Embedding layer that allows partial freezing of pretrained weights and addition of extra trainable embeddings.
    """

    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze=False,
        device=None,
        dtype=None,
        padding_idx=None,
        **kwargs,
    ) -> None:
        if padding_idx is not None and padding_idx > num_embeddings:
            raise ValueError(f"padding_idx must be within num_embeddings. Got {padding_idx} and {num_embeddings}")

        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=padding_idx,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        if partially_freeze:
            self.weight.requires_grad_(False)

        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )

    def forward(self, input_ids):
        if self.num_additional_embeddings == 0:
            return super().forward(input_ids)

        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)
        full_vector[additional_vocab_indices] = additional_embeddings
        return full_vector


@dataclass
class VBertBaseModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VBertMaskedLMOutput(MaskedLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class VBertSimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class VBertConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.pixel_shuffle_factor
        self.modality_projection = VBertSimpleMLP(
            input_size=config.vision_config.hidden_size * (config.scale_factor**2),
            output_size=config.text_config.hidden_size,
        )

    def pixel_shuffle(self, x, scale_factor):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        return x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        return self.modality_projection(image_hidden_states)


class VBertPreTrainedModel(PreTrainedModel):
    config_class = VBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VBertDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.text_config.initializer_range)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VBertModel(VBertPreTrainedModel):
    def __init__(self, config: VBertConfig, **kwargs):
        super().__init__(config)
        self.vision_model = VBertModel.init_vision_model(config, **kwargs)
        self.connector = VBertConnector(config)
        self.text_model = VBertModel.init_language_model(config, **kwargs)
        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.image_token_id = config.image_token_id
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.post_init()

    @staticmethod
    def init_vision_model(config: VBertConfig, **kwargs):
        vision_model_config = AutoConfig.from_pretrained(
            config.vision_config.vision_model_name,
            _attn_implementation=config._attn_implementation,
            torch_dtype=config.torch_dtype,
            **kwargs,
        )
        vision_model = AutoModel.from_config(vision_model_config, trust_remote_code=True, **kwargs)
        return getattr(vision_model, "vision_model", vision_model)

    @staticmethod
    def init_language_model(config: VBertConfig, **kwargs):
        text_model_config = AutoConfig.from_pretrained(
            config.text_config.text_model_name,
            _attn_implementation=config._attn_implementation,
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
            **kwargs,
        )
        text_model = AutoModel.from_config(text_model_config, trust_remote_code=True, **kwargs)
        embed_layer = DecoupledEmbedding(
            num_embeddings=text_model_config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_config["freeze_text_layers"],
            padding_idx=config.pad_token_id,
        )
        text_model.set_input_embeddings(embed_layer)
        return text_model

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(self, input_ids, inputs_embeds, image_hidden_states):
        _, patch_size, _ = image_hidden_states.shape
        image_mask = input_ids == self.image_token_id
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("Number of <image> tokens not divisible by patch_size.")
        blocks_per_sample = num_image_tokens // patch_size
        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx
        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]
        return torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        image_hidden_states=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)
        if pixel_values is not None:
            batch_size, num_images, _, _, _ = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            if not any(real_images_inds):
                real_images_inds[0] = True
            pixel_values = pixel_values[real_images_inds].contiguous()
            image_hidden_states = self.vision_model(pixel_values=pixel_values).last_hidden_state
            image_hidden_states = self.connector(image_hidden_states)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)
        if inputs_embeds is not None and image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(input_ids, inputs_embeds, image_hidden_states)
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)
        return VBertBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

class VBertLMHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        pretrained_config = AutoConfig.from_pretrained(config.text_config.text_model_name, trust_remote_code=True, **kwargs)
        pretrained_model = AutoModelForMaskedLM.from_config(pretrained_config, trust_remote_code=True, **kwargs)
        self.head = pretrained_model.head
        self.decoder = pretrained_model.decoder

    def forward(self, hidden_states):
        return self.decoder(self.head(hidden_states))


class VBertForMaskedLM(VBertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.image_token_id = config.image_token_id
        self.in_features = config.hidden_size
        self.out_additional_features = config.additional_vocab_size
        self.vocab_size = config.vocab_size
        self.model = VBertModel(config, **kwargs)
        self.lm_head = VBertLMHead(config, **kwargs)
        if self.out_additional_features > 0:
            self.additional_fc = nn.Linear(self.in_features, self.out_additional_features, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        pixel_values=None,
        image_hidden_states=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, VBertMaskedLMOutput]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_hidden_states=image_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.out_additional_features > 0:
            proj_states = self.lm_head.head(hidden_states)
            additional_features = self.additional_fc(proj_states)
            logits = torch.cat((logits, additional_features), -1)
        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits.view(-1, self.vocab_size + self.out_additional_features), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return VBertMaskedLMOutput(
            loss=loss,
            logits=logits.float(),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )