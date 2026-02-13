import re

from colpali_engine.models.paligemma.bipali.modeling_bipali import BiPali
from colpali_engine.models.paligemma.colpali.modeling_colpali import ColPali
from colpali_engine.models.qwen2.colqwen2.modeling_colqwen2 import ColQwen2
from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import ColQwen2_5


def _apply_mapping(key: str, mapping: dict[str, str]) -> str:
    remapped = key
    for pattern, replacement in mapping.items():
        remapped = re.sub(pattern, replacement, remapped)
    return remapped


def test_qwen2_adapter_key_mapping_remaps_custom_text_proj_and_layers():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColQwen2._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )
    assert (
        _apply_mapping(
            "model.layers.17.self_attn.v_proj.lora_B.default.weight",
            ColQwen2._checkpoint_conversion_mapping,
        )
        == "language_model.layers.17.self_attn.v_proj.lora_B.default.weight"
    )


def test_qwen2_5_adapter_key_mapping_remaps_custom_text_proj_and_layers():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_B.default.weight",
            ColQwen2_5._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_B.default.weight"
    )
    assert (
        _apply_mapping(
            "model.layers.3.mlp.down_proj.lora_A.default.weight",
            ColQwen2_5._checkpoint_conversion_mapping,
        )
        == "language_model.layers.3.mlp.down_proj.lora_A.default.weight"
    )


def test_colpali_adapter_key_mapping_remaps_custom_text_proj():
    assert (
        _apply_mapping(
            "base_model.model.custom_text_proj.lora_A.default.weight",
            ColPali._checkpoint_conversion_mapping,
        )
        == "custom_text_proj.lora_A.default.weight"
    )


def test_pali_wrappers_ignore_expected_missing_lm_head_weight():
    assert r"model\.lm_head\.weight" in ColPali._keys_to_ignore_on_load_missing
    assert r"model\.lm_head\.weight" in BiPali._keys_to_ignore_on_load_missing
