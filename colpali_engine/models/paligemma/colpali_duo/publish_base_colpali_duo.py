from typing import cast

import typer
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig

from colpali_engine.models.paligemma.colpali_duo.configuration_colpali_duo import ColPaliDuoConfig
from colpali_engine.models.paligemma.colpali_duo.modeling_colpali_duo import ColPaliDuo


def main():
    vlm_backbone_model_name = "google/paligemma-3b-mix-448"
    base_colpali_duo_name = "vidore/colpali-v2.0-test-0.1"

    model_config = ColPaliDuoConfig(
        vlm_backbone_config=cast(PaliGemmaConfig, PaliGemmaConfig.from_pretrained(vlm_backbone_model_name)),
        single_vector_projector_dim=1024,
        single_vector_pool_strategy="mean",
        multi_vector_projector_dim=128,
    )

    model = ColPaliDuo(config=model_config)

    model.push_to_hub(base_colpali_duo_name, private=True)

    return


if __name__ == "__main__":
    typer.run(main)