from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor


class ColPaliDuoProcessor(ColPaliProcessor):
    """
    Processor for ColPaliDuo. Mirrors the `ColPaliProcessor` class.

    TODO: duplicate transformers-ready code from https://github.com/huggingface/transformers/pull/33736 once merged
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
