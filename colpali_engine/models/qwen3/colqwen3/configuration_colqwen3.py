from copy import deepcopy
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig


class ColQwen3Config(PretrainedConfig):
    model_type = "colqwen3"
    sub_configs: dict[str, Any] = {
        "vision_config": Qwen3VLVisionConfig,
        "text_config": Qwen3VLTextConfig,
    }
    is_composition = True

    def __init__(
        self,
        vision_config: PretrainedConfig | dict[str, Any] | None = None,
        text_config: PretrainedConfig | dict[str, Any] | None = None,
        embed_dim: int = 320,
        padding_side: str = "left",
        initializer_range: float = 0.02,
        dtype: str | None = None,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Qwen3VLVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Qwen3VLVisionConfig(**deepcopy(vision_config))

        if text_config is None:
            text_config = Qwen3VLTextConfig()
        elif isinstance(text_config, dict):
            text_config = Qwen3VLTextConfig(**deepcopy(text_config))

        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.text_config = text_config
        self.embed_dim = embed_dim
        self.padding_side = padding_side
        self.initializer_range = initializer_range
        self.dtype = dtype or getattr(self, "dtype", None)

    def to_backbone_config(self) -> Qwen3VLConfig:
        config = Qwen3VLConfig(
            text_config=self.text_config.to_dict(),
            vision_config=self.vision_config.to_dict(),
            image_token_id=getattr(self, "image_token_id", 151655),
            video_token_id=getattr(self, "video_token_id", 151656),
            vision_start_token_id=getattr(self, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(self, "vision_end_token_id", 151653),
            tie_word_embeddings=getattr(self, "tie_word_embeddings", False),
        )

        for attr in ("dtype", "use_cache", "pad_token_id", "bos_token_id", "eos_token_id"):
            if hasattr(self, attr):
                setattr(config, attr, getattr(self, attr))

        return config
