from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor


class ColPali2Processor(ColPaliProcessor):
    """
    Processor for ColPali2. Mirrors the `ColPaliProcessor` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
