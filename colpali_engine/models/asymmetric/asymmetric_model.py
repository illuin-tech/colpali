from typing import ClassVar, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class AsymmetricModel(PreTrainedModel):
    """
    Base class for asymmetric models in the ColPali Engine.

    Args:
        config (PretrainedConfig): The model configuration.
        query_model (Optional[nn.Module]): Model used for encoding queries.
        document_model (Optional[nn.Module]): Model used for encoding documents.
    """

    config_class = PretrainedConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["Idefics3VisionAttention", "Idefics3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_attention_backend = True
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self,
        query_model: Optional[nn.Module] = None,
        document_model: Optional[nn.Module] = None,
        config: Optional[PretrainedConfig] = None,
    ):
        super().__init__(config=config)
        self.query_model = query_model or nn.Identity()
        self.document_model = document_model or nn.Identity()

        self.add_module("query_model", self.query_model)
        self.add_module("document_model", self.document_model)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the asymmetric model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The output of the selected sub-model.
        """
        if "pixel_values" in kwargs:
            return self.document_model(*args, **kwargs)
        return self.query_model(*args, **kwargs)


if __name__ == "__main__":
    import torch
    from PIL import Image

    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    from colpali_engine.models.asymmetric.asymmetric_model import AsymmetricModel

    query_model = ColIdefics3.from_pretrained(
        "vidore/colSmol-256M",
        torch_dtype=torch.float16,
        device_map="mps",
    ).eval()

    doc_model = ColIdefics3.from_pretrained(
        "vidore/colSmol-256M",
        torch_dtype=torch.float16,
        device_map="mps",
    ).eval()

    # Example usage
    config = PretrainedConfig()
    print(f"Query model parameters before: {sum(p.numel() for p in query_model.model.parameters())}")
    # Remove vision model in the query model
    query_model.model.vision_model = nn.Identity()

    # print num parameters in the query model
    print(f"Query model parameters after: {sum(p.numel() for p in query_model.model.parameters())}")
    # print num parameters in the document model
    print(f"Document model parameters: {sum(p.numel() for p in doc_model.model.parameters())}")

    model = AsymmetricModel(config=config, query_model=query_model, document_model=doc_model)
    model._set_static_graph()
    # print(model)

    processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-256M")

    # Your inputs
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (160, 160), color="black"),
    ]
    queries = ["This is black", "This is white", "This is red"]

    # Process the inputs
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    # Forward pass
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    print(scores)

    print(model)
    # # Save the model and processor
    # model.save_pretrained("asymmetric_model")
    # model.query_model.save_pretrained("asymmetric_query_model")
    # model.document_model.save_pretrained("asymmetric_document_model")
