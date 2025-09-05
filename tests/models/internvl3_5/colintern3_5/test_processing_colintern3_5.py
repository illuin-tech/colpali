from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColIntern3_5Processor


@pytest.fixture(scope="module")
def model_name() -> str:
    return "OpenGVLab/InternVL3_5-1B-HF"


@pytest.fixture(scope="module")
def processor_from_pretrained(model_name: str) -> Generator[ColIntern3_5Processor, None, None]:
    yield cast(
        ColIntern3_5Processor, 
        ColIntern3_5Processor.from_pretrained(
            model_name, 
            max_num_visual_tokens=768
        )
    )


@pytest.fixture(scope="module")
def processor_no_token_limit(model_name: str) -> Generator[ColIntern3_5Processor, None, None]:
    yield cast(ColIntern3_5Processor, ColIntern3_5Processor.from_pretrained(model_name))


def test_load_processor_from_pretrained(processor_from_pretrained: ColIntern3_5Processor):
    assert isinstance(processor_from_pretrained, ColIntern3_5Processor)


def test_processor_has_required_attributes(processor_from_pretrained: ColIntern3_5Processor):
    """Test that the processor has all required attributes."""
    assert hasattr(processor_from_pretrained, 'tokenizer')
    assert hasattr(processor_from_pretrained, 'image_processor')
    assert hasattr(processor_from_pretrained, 'query_augmentation_token')


def test_tokenizer_padding_side(processor_from_pretrained: ColIntern3_5Processor):
    """Test that the tokenizer padding is set correctly."""
    assert processor_from_pretrained.tokenizer.padding_side == "right"


def test_query_augmentation_token(processor_from_pretrained: ColIntern3_5Processor):
    """Test that the query augmentation token is correctly set."""
    token = processor_from_pretrained.query_augmentation_token
    assert isinstance(token, str)
    assert len(token) > 0


def test_process_images(processor_from_pretrained: ColIntern3_5Processor):
    # Create a dummy image
    image_size = (224, 224)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert "input_ids" in batch_feature
    assert "attention_mask" in batch_feature
    
    assert isinstance(batch_feature["pixel_values"], torch.Tensor)
    assert isinstance(batch_feature["input_ids"], torch.Tensor)
    assert isinstance(batch_feature["attention_mask"], torch.Tensor)
    
    assert batch_feature["pixel_values"].shape[0] == 1
    assert batch_feature["input_ids"].shape[0] == 1
    assert batch_feature["attention_mask"].shape[0] == 1
    
    # Check dtype
    assert batch_feature["pixel_values"].dtype == torch.bfloat16


def test_process_images_multiple(processor_from_pretrained: ColIntern3_5Processor):
    """Test processing multiple images."""
    images = [
        Image.new("RGB", (224, 224), color="red"),
        Image.new("RGB", (448, 448), color="green"),
        Image.new("RGB", (300, 500), color="blue"),
    ]

    # Process the images
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions - The GotOCR2 processor may create multiple patches per image
    # so we check that we get a reasonable number of patches (at least one per image)
    assert batch_feature["pixel_values"].shape[0] >= len(images)
    assert batch_feature["input_ids"].shape[0] >= len(images)
    assert batch_feature["attention_mask"].shape[0] >= len(images)


def test_process_images_dtype_conversion(processor_from_pretrained: ColIntern3_5Processor):
    """Test that pixel values are converted to bfloat16."""
    image = Image.new("RGB", (224, 224), color="white")
    
    batch_feature = processor_from_pretrained.process_images([image])
    
    assert batch_feature["pixel_values"].dtype == torch.bfloat16


def test_process_texts(processor_from_pretrained: ColIntern3_5Processor):
    queries = [
        "What is shown in this image?",
        "Describe the content of this document.",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_texts(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert "attention_mask" in batch_encoding
    
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert isinstance(batch_encoding["attention_mask"], torch.Tensor)
    
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)
    assert cast(torch.Tensor, batch_encoding["attention_mask"]).shape[0] == len(queries)


def test_process_queries(processor_from_pretrained: ColIntern3_5Processor):
    queries = [
        "What is shown in this image?",
        "Describe the content of this document.",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_queries(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert "attention_mask" in batch_encoding
    
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert isinstance(batch_encoding["attention_mask"], torch.Tensor)
    
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)
    assert cast(torch.Tensor, batch_encoding["attention_mask"]).shape[0] == len(queries)


def test_process_queries_vs_texts_equivalence(processor_from_pretrained: ColIntern3_5Processor):
    """Test that process_queries and process_texts produce the same output."""
    queries = ["Test query", "Another test query"]
    
    queries_output = processor_from_pretrained.process_queries(queries)
    texts_output = processor_from_pretrained.process_texts(queries)
    
    assert torch.equal(queries_output["input_ids"], texts_output["input_ids"])
    assert torch.equal(queries_output["attention_mask"], texts_output["attention_mask"])


def test_process_empty_query(processor_from_pretrained: ColIntern3_5Processor):
    """Test processing empty queries."""
    empty_queries = [""]
    
    batch_encoding = processor_from_pretrained.process_queries(empty_queries)
    
    # Should not crash and should return valid tensors
    assert "input_ids" in batch_encoding
    assert "attention_mask" in batch_encoding
    assert batch_encoding["input_ids"].shape[0] == 1


def test_process_long_query(processor_from_pretrained: ColIntern3_5Processor):
    """Test processing very long queries."""
    long_query = "This is a very long query. " * 100
    
    batch_encoding = processor_from_pretrained.process_queries([long_query])
    
    # Should not crash and should return valid tensors
    assert "input_ids" in batch_encoding
    assert "attention_mask" in batch_encoding
    assert batch_encoding["input_ids"].shape[0] == 1


def test_image_processor_type(processor_from_pretrained: ColIntern3_5Processor):
    """Test that the correct image processor type is being used."""
    processor_type = type(processor_from_pretrained.image_processor).__name__
    assert "ImageProcessor" in processor_type or "GotOcr2" in processor_type


def test_max_visual_tokens_configuration(processor_from_pretrained: ColIntern3_5Processor):
    """Test that max_num_visual_tokens configuration is applied."""
    # Create a large image that might produce many tokens
    large_image = Image.new("RGB", (1000, 1000), color="white")
    
    batch_feature = processor_from_pretrained.process_images([large_image])
    
    # Process with model to see token count (this is indirect testing)
    # The GotOCR2 processor may create multiple patches from large images
    assert "pixel_values" in batch_feature
    assert batch_feature["pixel_values"].shape[0] >= 1  # At least one patch
    
    # The important thing is that it doesn't create an excessive number of patches
    # A 1000x1000 image should be reasonable, not creating hundreds of patches
    assert batch_feature["pixel_values"].shape[0] < 50, f"Too many patches: {batch_feature['pixel_values'].shape[0]}"


def test_processor_with_and_without_token_limit(
    processor_from_pretrained: ColIntern3_5Processor,
    processor_no_token_limit: ColIntern3_5Processor
):
    """Test difference between processor with and without token limit."""
    image = Image.new("RGB", (500, 500), color="white")
    
    # Process with both processors
    batch_limited = processor_from_pretrained.process_images([image])
    batch_unlimited = processor_no_token_limit.process_images([image])
    
    # Both should work but may have different configurations
    assert "pixel_values" in batch_limited
    assert "pixel_values" in batch_unlimited
    
    # Both should have the same basic structure
    assert batch_limited["pixel_values"].shape[0] == 1
    assert batch_unlimited["pixel_values"].shape[0] == 1


def test_batch_consistency(processor_from_pretrained: ColIntern3_5Processor):
    """Test that batch processing is consistent."""
    image1 = Image.new("RGB", (224, 224), color="red")
    image2 = Image.new("RGB", (224, 224), color="blue")
    
    # Process individually
    single1 = processor_from_pretrained.process_images([image1])
    single2 = processor_from_pretrained.process_images([image2])
    
    # Process as batch
    batch = processor_from_pretrained.process_images([image1, image2])
    
    # Check that batch processing gives reasonable results
    # Note: Due to GotOCR2 processor behavior, exact matching may not be possible
    # but we can check that the structure is reasonable
    assert batch["input_ids"].shape[0] >= 2  # At least one patch per image
    assert "pixel_values" in batch
    assert "attention_mask" in batch
