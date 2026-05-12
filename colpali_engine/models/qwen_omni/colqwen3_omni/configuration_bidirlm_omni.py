from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


# ── Audio encoder config ──────────────────────────────


class BidirLMOmniAudioConfig(PretrainedConfig):
    model_type = "bidirlm_omni_audio"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


# ── Vision encoder config ─────────────────────────────


class BidirLMOmniVisionConfig(PretrainedConfig):
    model_type = "bidirlm_omni_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = [8, 16, 24]
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes


# ── Shared text encoder config ──────────────────────────────────────────────


class BidirLMOmniTextConfig(PretrainedConfig):
    model_type = "bidirlm_omni_text"
    base_config_key = "text_config"
    # mrope_section/mrope_interleaved are model-specific rope_scaling keys.
    # Without this, validate_rope() called by huggingface_hub warns about them.
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        clf_pooling="late",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.clf_pooling = clf_pooling
        self.is_causal = False

        # In tf5, super().__init__() calls convert_rope_params_to_dict() + validate_rope()
        # automatically via huggingface_hub. ignore_keys_at_rope_validation (class attr above)
        # tells validate_rope() to skip mrope_section/mrope_interleaved warnings.
        # The old rope_config_validation() call is not needed and emits a FutureWarning.
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# ── Top-level omni config ──────────────────────────────────────────────────


class BidirLMOmniConfig(PretrainedConfig):
    model_type = "bidirlm_omni"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}
    sub_configs = {
        "audio_config": BidirLMOmniAudioConfig,
        "vision_config": BidirLMOmniVisionConfig,
        "text_config": BidirLMOmniTextConfig,
    }

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        vision_config=None,
        # Audio special tokens
        audio_token_id=151676,
        audio_start_token_id=151669,
        audio_end_token_id=151670,
        # Vision special tokens
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
        text_weights_source="visual",
        # Classification / fine-tuning
        num_labels=1,
        problem_type=None,
        clf_pooling="late",
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            self.audio_config = BidirLMOmniAudioConfig(**audio_config)
        elif audio_config is None:
            self.audio_config = BidirLMOmniAudioConfig()
        else:
            self.audio_config = audio_config

        if isinstance(vision_config, dict):
            self.vision_config = BidirLMOmniVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = BidirLMOmniVisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = BidirLMOmniTextConfig(**text_config)
        elif text_config is None:
            self.text_config = BidirLMOmniTextConfig()
        else:
            self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.text_weights_source = text_weights_source
        self.clf_pooling = clf_pooling

        # num_labels / problem_type must be set AFTER super().__init__() because
        # PretrainedConfig.num_labels is a property that accesses id2label, which
        # is only initialised by super().__init__().
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.num_labels = num_labels
        self.problem_type = problem_type


__all__ = [
    "BidirLMOmniConfig",
    "BidirLMOmniTextConfig",
    "BidirLMOmniAudioConfig",
    "BidirLMOmniVisionConfig",
]
