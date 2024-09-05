from transformers import PretrainedConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaConfig


class ColPali2Config(PretrainedConfig):
    def __init__(
        self,
        vlm_config: PaliGemmaConfig,
        single_vector_projector_dim: int = 128,
        single_vector_pool_strategy: str = "mean",
        multi_vector_projector_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vlm_config = vlm_config
        self.single_vector_projector_dim = single_vector_projector_dim
        self.single_vector_pool_strategy = single_vector_pool_strategy
        self.multi_vector_projector_dim = multi_vector_projector_dim
