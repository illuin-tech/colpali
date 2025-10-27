from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, GenerationMixin, logging
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs

# from transformers.models.smolvlm import SmolVLMModel, SmolVLMPreTrainedModel
from .configuration_vllama import VLlamaConfig

logger = logging.get_logger(__name__)

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings.
    In practise, the regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0, then it will create `num_additional_embeddings` additional parameters that are always trained.
    If `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
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
        """
        num_additional_embeddings: int. Number of additional embeddings. Only useful when you `partially_freeze=True`.
        partially_freeze: bool. If True, the regular `weight` will be frozen. `additional_weight` is never frozen.
        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`, `max_norm` or `norm_type`. We are not supporting these.
        """
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
        self.padding_idx = padding_idx
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        if partially_freeze:
            self.weight.requires_grad_(False)

        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.
        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings),
           since the 2nd embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup
        note: for the 1st embedding lookup we could have looked up only the low indices and not do
        the padding, but then we have to create a new tensor and populate it with 2 tensors that are
        spread out across various indices - i.e. not a simple concat - I haven't benchmarked the
        complex case if it's any faster, given that seqlens are usually relatively short it's
        probably not faster or if faster not by much - but might be a good idea to measure.
        """
        if self.num_additional_embeddings == 0:
            return self.additional_embedding(input_ids)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.num_embeddings,
            self.num_additional_embeddings,
            self.embedding_dim,
            self.partially_freeze,
        )

@dataclass
class VLlamaBaseModelOutputWithPast(BaseModelOutput):
    """
    Base class for VLlama3 model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class VLlamaCausalLMOutputWithPast(BaseModelOutput):
    """
    Base class for VLlama3 causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class VLlamaSimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)

class VLlamaConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.pixel_shuffle_factor
        self.modality_projection = VLlamaSimpleMLP(
            input_size=config.vision_config.hidden_size * (config.scale_factor**2),
            output_size=config.text_config.hidden_size
        )

    def pixel_shuffle(self, x, scale_factor):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states

class VLlamaPreTrainedModel(PreTrainedModel):
    config_class = VLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        """Initialize the weights."""

        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class VLlamaModel(VLlamaPreTrainedModel):
    """
    A subclass of Idefics3Model. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """

    def __init__(self, config: VLlamaConfig, **kwargs):
        super().__init__(config)

        self.vision_model = VLlamaModel.init_vision_model(config, **kwargs)
        self.connector = VLlamaConnector(config)
        self.text_model = VLlamaModel.init_language_model(config, **kwargs)

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.post_init()

    @staticmethod
    def init_vision_model(config: VLlamaConfig, **kwargs):
        vision_model_config = AutoConfig.from_pretrained(
            config.vision_config.vision_model_name,
            _attn_implementation=config._attn_implementation,
            torch_dtype=config.torch_dtype,
            **kwargs,
        )

        vision_model = AutoModel.from_config(vision_model_config, trust_remote_code=True, **kwargs)

        if hasattr(vision_model, "vision_model"):
            # If the model has a vision_model attribute, it means it's a wrapper around another model
            vision_model = vision_model.vision_model

        return vision_model

    @staticmethod
    def init_language_model(config: VLlamaConfig, **kwargs):
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

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.
        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032
        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        _, patch_size, _ = image_hidden_states.shape

        image_mask = input_ids == self.image_token_id
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]

        merged_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)
        return merged_embeds

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Override the embed_tokens method to use the text model's input embeddings.
        This is necessary to ensure that the image token ID is correctly handled.
        """
        if self.text_model.get_input_embeddings() is None:
            raise ValueError("The text model does not have input embeddings.")

        return self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, VLlamaBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if inputs_embeds is not None and input_ids is None:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

            if not any(real_images_inds):
                # no images, leave one empty image.
                real_images_inds[0] = True

            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=[pixel_values.shape[i] for i in (0, 2, 3)],
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            # patch_size = self.config.vision_config.patch_size
            # patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            # patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            # patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                # patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if inputs_embeds is not None and image_hidden_states is not None:
            # When we embed, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return VLlamaBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

class VLlamaForCausalLM(VLlamaPreTrainedModel):
    # _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.image_token_id = config.image_token_id
        self.in_features = config.hidden_size
        self.out_additional_features = config.additional_vocab_size
        self.vocab_size = config.vocab_size

        self.model = VLlamaModel(config, **kwargs)
        self.lm_head = VLlamaForCausalLM.init_lm_head(config, **kwargs)
        if self.out_additional_features > 0:
            self.additional_fc = nn.Linear(
                in_features=self.in_features,
                out_features=self.out_additional_features,
                bias=False,
            )

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def init_lm_head(config, **kwargs):
        # Get the pretrained model config
        text_model_config = AutoConfig.from_pretrained(
            config.text_config.text_model_name,
            trust_remote_code=True,
            **kwargs,
        )
        model = AutoModelForMaskedLM.from_config(text_model_config, trust_remote_code=True, **kwargs)
        # Get the lm head
        lm_head = model.lm_head if hasattr(model, "lm_head") else model.decoder if hasattr(model, "decoder") else None
        if lm_head is None:
            logger.warning(f"No lm head was found for {config.text_config.text_model_name}, initializing a new one.")
            lm_head = nn.Linear(config.hidden_size, config.vocab_size, False)
        return lm_head

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pixel_attention_mask: Optional[torch.BoolTensor] = None,
            image_hidden_states: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, VLlamaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics3ForConditionalGeneration`).
                Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
                computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # Pass the inputs to VLlamaModel
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # Pass the outputs to the MLM head
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        if self.out_additional_features > 0:
            additional_features = self.additional_fc(hidden_states)
            logits = torch.cat((logits, additional_features), -1)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VLlamaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

class VLlamaForVision2Seq(VLlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.image_token_id = config.image_token_id
        self.in_features = config.hidden_size
        self.out_additional_features = config.additional_vocab_size
        self.vocab_size = config.vocab_size

        self.model = VLlamaModel(config, **kwargs)
        self.lm_head = VLlamaForVision2Seq.init_lm_head(config, **kwargs)
        if self.out_additional_features > 0:
            self.additional_fc = nn.Linear(
                in_features=self.in_features,
                out_features=self.out_additional_features,
                bias=False,
            )

        self.loss_fct = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def init_lm_head(config, **kwargs):
        # Get the pretrained model config
        text_model_config = AutoConfig.from_pretrained(
            config.text_config.text_model_name,
            trust_remote_code=True,
            **kwargs,
        )
        model = AutoModelForMaskedLM.from_config(text_model_config, trust_remote_code=True, **kwargs)
        # Get the lm head
        lm_head = model.lm_head if hasattr(model, "lm_head") else model.decoder if hasattr(model, "decoder") else None
        if lm_head is None:
            logger.warning(f"No lm head was found for {config.text_config.text_model_name}, initializing a new one.")
            lm_head = nn.Linear(config.hidden_size, config.vocab_size, False)
        return lm_head

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = self.model.vision_model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.model.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, VLlamaCausalLMOutputWithPast]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `SmolVLMForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        >>> model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

        >>> # Create inputs
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "video", "path": path/to/video},
        ...             {"type": "text", "text": "What is happening in this video?"},
        ...         ]
        ...     }
        ... ]

        >>> inputs = processor.apply_chat_template([messages], add_generation_prompt=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=256)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states)
        if self.out_additional_features > 0:
            additional_features = self.additional_fc(hidden_states)
            logits = torch.cat((logits, additional_features), -1)
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return VLlamaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- there are mutually exclusive inputs (if the logic to make `image_hidden_states` take
        # precedence is moved to the model, we can remove this fn)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # but IDEFICS requires both ids and embeds to be present
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs["input_ids"] = input_ids

        if image_hidden_states is not None:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        # Get the precomputed image_hidden_states
        model_kwargs["image_hidden_states"] = outputs.image_hidden_states
        return model_kwargs

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
