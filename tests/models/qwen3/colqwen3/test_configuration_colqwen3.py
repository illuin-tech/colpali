from transformers.models.qwen3_vl import Qwen3VLConfig

from colpali_engine.models.qwen3.colqwen3 import ColQwen3, ColQwen3Config


def test_colqwen3_config_converts_to_qwen3_vl_config():
    config = ColQwen3Config(
        text_config={"hidden_size": 64, "num_hidden_layers": 3},
        vision_config={"hidden_size": 32, "depth": 2},
        embed_dim=24,
        image_token_id=11,
        video_token_id=12,
        vision_start_token_id=13,
        vision_end_token_id=14,
    )

    backbone_config = config.to_backbone_config()

    assert isinstance(backbone_config, Qwen3VLConfig)
    assert backbone_config.text_config.hidden_size == 64
    assert backbone_config.text_config.num_hidden_layers == 3
    assert backbone_config.vision_config.hidden_size == 32
    assert backbone_config.vision_config.depth == 2
    assert backbone_config.image_token_id == 11
    assert backbone_config.video_token_id == 12
    assert backbone_config.vision_start_token_id == 13
    assert backbone_config.vision_end_token_id == 14


def test_colqwen3_prefers_text_hidden_size_over_top_level_hidden_size():
    config = Qwen3VLConfig(
        text_config={"hidden_size": 64},
        vision_config={"hidden_size": 32},
    )
    config.hidden_size = 32

    assert ColQwen3._get_text_hidden_size(config) == 64
