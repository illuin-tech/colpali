from transformers.models.paligemma import PaliGemmaConfig


class ColPaliDuoConfig(PaliGemmaConfig):
    """
    Configuration for the ColPaliDuo model.
    """

    def __init__(
        self,
        single_vector_projector_dim: int = 1024,
        single_vector_pool_strategy: str = "mean",
        multi_vector_projector_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.single_vector_projector_dim = single_vector_projector_dim
        self.single_vector_pool_strategy = single_vector_pool_strategy
        self.multi_vector_projector_dim = multi_vector_projector_dim
