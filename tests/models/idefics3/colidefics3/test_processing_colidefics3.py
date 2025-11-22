from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColIdefics3Processor


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colSmol-256M"


@pytest.fixture(scope="module")
def processor_from_pretrained(
    model_name: str,
) -> Generator[ColIdefics3Processor, None, None]:
    yield cast(ColIdefics3Processor, ColIdefics3Processor.from_pretrained(model_name))


def test_load_processor_from_pretrained(
    processor_from_pretrained: ColIdefics3Processor,
):
    assert isinstance(processor_from_pretrained, ColIdefics3Processor)


def test_process_images(processor_from_pretrained: ColIdefics3Processor):
    # Create a dummy image
    image_size = (16, 32)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert batch_feature["pixel_values"].shape == torch.Size([1, 9, 3, 512, 512])


def test_process_texts(processor_from_pretrained: ColIdefics3Processor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the texts
    batch_encoding = processor_from_pretrained.process_texts(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)


def test_process_queries(processor_from_pretrained: ColIdefics3Processor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_queries(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)


def test_get_n_patches(processor_from_pretrained: ColIdefics3Processor):
    """
    Test that get_n_patches returns the correct number of patches for various image sizes.
    """
    # Get the patch size from the image processor
    patch_size = processor_from_pretrained.image_processor.max_image_size.get(
        "longest_edge", 512
    )

    # Test case 1: Small square image
    image_size = (100, 100)
    n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
        image_size, patch_size
    )
    assert isinstance(n_patches_x, int)
    assert isinstance(n_patches_y, int)
    assert n_patches_x > 0
    assert n_patches_y > 0

    # Test case 2: Wide image (width > height)
    image_size = (100, 200)
    n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
        image_size, patch_size
    )
    assert n_patches_x >= n_patches_y  # More patches along width

    # Test case 3: Tall image (height > width)
    image_size = (200, 100)
    n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
        image_size, patch_size
    )
    assert n_patches_y >= n_patches_x  # More patches along height

    # Test case 4: Square image
    image_size = (500, 500)
    n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
        image_size, patch_size
    )
    assert n_patches_x == n_patches_y  # Equal patches for square image


def test_get_n_patches_matches_actual_processing(
    processor_from_pretrained: ColIdefics3Processor,
):
    """
    Test that get_n_patches matches the actual number of patches produced by process_images.
    """
    # Create a test image
    image_size = (16, 32)  # PIL Image.new takes (width, height)
    image = Image.new("RGB", image_size, color="black")

    # Process the image to get actual patch count
    batch_feature = processor_from_pretrained.process_images([image])
    # pixel_values shape is [batch_size, num_patches, channels, patch_height, patch_width]
    actual_num_patches = batch_feature["pixel_values"].shape[1]

    # Get the patch size from the image processor
    patch_size = processor_from_pretrained.image_processor.max_image_size.get(
        "longest_edge", 512
    )

    # Calculate expected patches using get_n_patches
    # Note: image_size for get_n_patches is (height, width), but PIL uses (width, height)
    n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
        (image_size[1], image_size[0]), patch_size
    )
    expected_num_patches = n_patches_x * n_patches_y

    # The actual number of patches includes the global image patch (+1)
    # So we compare with expected + 1
    assert (
        actual_num_patches == expected_num_patches + 1
    ), f"Expected {expected_num_patches + 1} patches (including global), got {actual_num_patches}"


def test_get_image_mask(processor_from_pretrained: ColIdefics3Processor):
    """
    Test that get_image_mask correctly identifies image tokens.
    """
    # Create a dummy image
    image_size = (16, 32)
    image = Image.new("RGB", image_size, color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Get the image mask
    image_mask = processor_from_pretrained.get_image_mask(batch_feature)

    # Assertions
    assert isinstance(image_mask, torch.Tensor)
    assert image_mask.shape == batch_feature.input_ids.shape
    assert image_mask.dtype == torch.bool
    # There should be some image tokens (True values) in the mask
    assert image_mask.sum() > 0
