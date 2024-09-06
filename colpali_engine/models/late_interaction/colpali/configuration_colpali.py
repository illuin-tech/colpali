from transformers.models.paligemma.modeling_paligemma import PaliGemmaConfig


class ColPaliConfig(PaliGemmaConfig):
    def __init__(
        self,
        embedding_dim: int = 128,
        **kwargs,
    ):
        """
        Config for ColPali model.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
