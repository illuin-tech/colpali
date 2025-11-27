from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColIdefics3Processor


@pytest.fixture(scope="module")
def model_name() -> str:
    return "vidore/colSmol-256M"


@pytest.fixture(scope="module")
def processor_from_pretrained(model_name: str) -> Generator[ColIdefics3Processor, None, None]:
    yield cast(ColIdefics3Processor, ColIdefics3Processor.from_pretrained(model_name))


def test_load_processor_from_pretrained(processor_from_pretrained: ColIdefics3Processor):
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


def test_get_n_patches_with_resize_enabled(processor_from_pretrained: ColIdefics3Processor):
    """Test get_n_patches with default do_resize=True setting."""
    # Test with a standard image size
    image_size = (800, 600)
    n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(image_size)

    # Expected values based on the resizing pipeline:
    # (800, 600) -> resize to (1456, 1092) -> scale to (1456, 1092) -> align to (1456, 1092)
    # Grid cells: 1456/364 = 4, 1092/364 = 3
    # Patches: 4*8 = 32, 3*8 = 24 (where 8 = sqrt(image_seq_len))
    assert n_patches_x == 32
    assert n_patches_y == 24


@pytest.mark.parametrize("do_resize", [True, False])
def test_get_n_patches_with_different_resize_settings(
    processor_from_pretrained: ColIdefics3Processor,
    do_resize: bool,
):
    """Test get_n_patches with both do_resize=True and do_resize=False."""
    # Store original setting for cleanup
    original_do_resize = processor_from_pretrained.image_processor.do_resize

    try:
        # Set the do_resize parameter
        processor_from_pretrained.image_processor.do_resize = do_resize

        # Test with a standard image size
        image_size = (800, 600)
        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(image_size)

        # Verify expected values based on do_resize setting
        if do_resize:
            # With resize: (800,600) -> (1456,1092) -> grid (4,3) -> patches (32,24)
            assert n_patches_x == 32
            assert n_patches_y == 24
        else:
            # Without resize: (800,600) -> align to (728,728) -> grid (2,2) -> patches (16,16)
            assert n_patches_x == 16
            assert n_patches_y == 16
    finally:
        # Restore original setting
        processor_from_pretrained.image_processor.do_resize = original_do_resize


@pytest.mark.parametrize(
    "image_size,expected_patches_with_resize",
    [
        ((64, 32), (32, 16)),  # Very small landscape
        ((32, 64), (16, 32)),  # Very small portrait
        ((364, 364), (32, 32)),  # Single grid cell size
        ((800, 600), (32, 24)),  # Standard landscape
        ((600, 800), (24, 32)),  # Standard portrait
        ((1024, 1024), (32, 32)),  # Large square
    ],
)
def test_get_n_patches_various_image_sizes(
    processor_from_pretrained: ColIdefics3Processor,
    image_size: tuple[int, int],
    expected_patches_with_resize: tuple[int, int],
):
    """Test get_n_patches with various image sizes."""
    # Ensure do_resize is enabled (default behavior)
    original_do_resize = processor_from_pretrained.image_processor.do_resize
    processor_from_pretrained.image_processor.do_resize = True

    try:
        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(image_size)
        expected_x, expected_y = expected_patches_with_resize
        assert n_patches_x == expected_x, f"Expected {expected_x} patches in x, got {n_patches_x}"
        assert n_patches_y == expected_y, f"Expected {expected_y} patches in y, got {n_patches_y}"
    finally:
        processor_from_pretrained.image_processor.do_resize = original_do_resize
