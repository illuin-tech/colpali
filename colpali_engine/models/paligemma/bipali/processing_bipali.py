from __future__ import annotations

from typing import Optional

from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor


class BiPaliProcessor(ColPaliProcessor):
    """
    Processor for BiPali. Mirrors the `ColPaliProcessor` class.
    """

    def __init__(
        self,
        vlm_backbone_model_name_or_path: str = "google/paligemma-3b-mix-448",
        hf_token: Optional[str] = None,
    ):
        super().__init__(vlm_backbone_model_name_or_path, hf_token)
