from __future__ import annotations

from colpali_engine.models.idefics_2.colidefics_2.processing_colidefics_2 import ColIdefics2Processor


class BiIdefics2Processor(ColIdefics2Processor):
    """
    Processor for BiIdefics2. Mirrors the `ColIdefics2Processor` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
