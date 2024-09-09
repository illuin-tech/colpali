from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor


class BiPaliProcessor(ColPaliProcessor):
    """
    Processor for BiPali. Mirrors the `ColPaliProcessor` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
