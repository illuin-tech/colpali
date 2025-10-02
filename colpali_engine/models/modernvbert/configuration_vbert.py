import copy
import os
from typing import Any, Dict, Union

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

def collect_arg_in_candidates(config, candidates, default=None) -> Any:
    """Gets the first available argument in a config given a list of candidate names."""
    for c in candidates:
        if hasattr(config, c):
            return getattr(config, c)
        elif c in config:
            return config[c]
    if default is not None:
        return default
    raise ValueError(
        f"No matching arguments found in candidates. Candidates: {candidates}, Config: {config}"
    )

class VBertTextConfig(PretrainedConfig):
    model_type = "vbert"

    def __init__(
        self,
        text_model_name="ModernVBERT/modernvbert",
        **kwargs,
    ):
        self.text_model_name = text_model_name
        text_config = AutoConfig.from_pretrained(text_model_name, trust_remote_code=True)
        if hasattr(text_config, "text_config"):
            text_config = text_config.text_config

        self.hidden_size = collect_arg_in_candidates(text_config, ["hidden_size", "embed_dim"])
        self.num_hidden_layers = collect_arg_in_candidates(text_config, ["num_hidden_layers", "num_hidden_blocks"])
        self.intermediate_size = collect_arg_in_candidates(text_config, ["intermediate_size", "mlp_dim"])
        self.mlp_bias = collect_arg_in_candidates(text_config, ["mlp_bias", "mlp_hidden_bias"], default=False)
        self.vocab_size = collect_arg_in_candidates(text_config, ["vocab_size"])

        super().__init__(text_model_name=text_model_name, **kwargs)

class VBertVisionConfig(PretrainedConfig):
    model_type = "vbert"
    attribute_map = {
        "hidden_size": "embed_dim",
    }

    def __init__(
        self,
        vision_model_name="google/siglip2-base-patch16-512",
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        vision_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config

        self.embed_dim = collect_arg_in_candidates(vision_config, ["embed_dim", "hidden_size"])
        self.image_size = collect_arg_in_candidates(vision_config, ["image_size", "img_size"])
        self.patch_size = collect_arg_in_candidates(vision_config, ["patch_size"])
        self.num_hidden_layers = collect_arg_in_candidates(vision_config, ["num_hidden_layers", "num_hidden_blocks"])
        self.intermediate_size = collect_arg_in_candidates(vision_config, ["intermediate_size", "mlp_dim"])

        super().__init__(vision_model_name=vision_model_name, **kwargs)

class VBertConfig(PretrainedConfig):
    model_type = "vbert"
    is_composition = True

    DEFAULT_TEXT_MODEL_NAME = "ModernVBERT/modernvbert"
    DEFAULT_VISION_MODEL_NAME = "google/siglip2-base-patch16-512"

    def __init__(
        self,
        text_config: Union[PretrainedConfig, Dict[str, Any]] = None,
        vision_config: Union[PretrainedConfig, Dict[str, Any]] = None,
        image_token_id: int = 128_257,
        vocab_size=128_256,
        use_cache=True,
        tie_word_embeddings=False,
        freeze_config=None,
        pad_token_id=None,
        initializer_range=0.02,
        pixel_shuffle_factor=4,
        use_resampler=False,
        additional_vocab_size=0,
        neftune_noise_alpha=0.0,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.scale_factor = pixel_shuffle_factor
        self.additional_vocab_size = additional_vocab_size

        if text_config is None:
            text_config = AutoConfig.from_pretrained(self.DEFAULT_TEXT_MODEL_NAME, trust_remote_code=True)
        elif isinstance(text_config, dict):
            text_config = VBertTextConfig(text_config["text_model_name"])
        self.text_config = text_config

        if vision_config is None:
            vision_config = AutoConfig.from_pretrained(self.DEFAULT_VISION_MODEL_NAME, trust_remote_code=True)
        elif isinstance(vision_config, dict):
            vision_config = VBertVisionConfig(vision_config["vision_model_name"])
        self.vision_config = vision_config

        self.freeze_config = freeze_config
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.use_resampler = use_resampler
        self.neftune_noise_alpha = neftune_noise_alpha
        self.initializer_range = initializer_range

        hidden_size = kwargs.pop("hidden_size", self.text_config.hidden_size)

        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        return output

    @classmethod
    def from_pretrained_models(
        cls,
        text_model_name: Union[str, os.PathLike],
        vision_model_name: Union[str, os.PathLike],
        **kwargs,
    ) -> "PretrainedConfig":
        text_model_config = VBertTextConfig(text_model_name)
        vision_model_config = VBertVisionConfig(vision_model_name)
        return cls(
            text_config=text_model_config,
            vision_config=vision_model_config,
            **kwargs,
        )