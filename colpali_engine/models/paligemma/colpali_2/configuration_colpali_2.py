from transformers import PretrainedConfig


class ColPali2Config(PretrainedConfig):
    """
    Configuration for the ColPali2 model.
    """

    def __init__(
        self,
        vlm_backbone_config: PretrainedConfig,
        single_vector_projector_dim: int = 1024,
        single_vector_pool_strategy: str = "mean",
        multi_vector_projector_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vlm_backbone_config = vlm_backbone_config
        self.single_vector_projector_dim = single_vector_projector_dim
        self.single_vector_pool_strategy = single_vector_pool_strategy
        self.multi_vector_projector_dim = multi_vector_projector_dim
