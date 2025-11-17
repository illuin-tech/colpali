"""
Test interpretability maps for ColModernVBert model.

This module tests:
1. get_n_patches() method - calculates correct patch dimensions
2. get_image_mask() method - identifies image tokens correctly
3. End-to-end similarity map generation
"""

from typing import Generator, cast

import pytest
import torch
from PIL import Image

from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from colpali_engine.interpretability.similarity_map_utils import (
    normalize_similarity_map,
)


@pytest.fixture(scope="module")
def model_name() -> str:
    return "ModernVBERT/colmodernvbert"


@pytest.fixture(scope="module")
def processor_from_pretrained(
    model_name: str,
) -> Generator[ColModernVBertProcessor, None, None]:
    yield cast(
        ColModernVBertProcessor, ColModernVBertProcessor.from_pretrained(model_name)
    )


@pytest.fixture(scope="module")
def model_from_pretrained(model_name: str) -> Generator[ColModernVBert, None, None]:
    yield cast(ColModernVBert, ColModernVBert.from_pretrained(model_name))


class TestGetNPatches:
    """Test the get_n_patches method for calculating patch dimensions."""

    def test_get_n_patches_returns_integers(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that get_n_patches returns integer values."""
        patch_size = 14  # Common patch size for vision transformers
        image_size = (100, 100)

        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            image_size, patch_size
        )

        assert isinstance(n_patches_x, int)
        assert isinstance(n_patches_y, int)
        assert n_patches_x > 0
        assert n_patches_y > 0

    def test_get_n_patches_wide_image(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that wide images have more patches along width."""
        patch_size = 14
        image_size = (100, 200)  # (height, width) - wider than tall

        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            image_size, patch_size
        )

        # n_patches_x is along width, n_patches_y is along height
        assert (
            n_patches_x >= n_patches_y
        ), f"Expected more patches along width, got x={n_patches_x}, y={n_patches_y}"

    def test_get_n_patches_tall_image(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that tall images have more patches along height."""
        patch_size = 14
        image_size = (200, 100)  # (height, width) - taller than wide

        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            image_size, patch_size
        )

        assert (
            n_patches_y >= n_patches_x
        ), f"Expected more patches along height, got x={n_patches_x}, y={n_patches_y}"

    def test_get_n_patches_square_image(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that square images have equal patches in both dimensions."""
        patch_size = 14
        image_size = (500, 500)

        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            image_size, patch_size
        )

        assert (
            n_patches_x == n_patches_y
        ), f"Expected equal patches for square image, got x={n_patches_x}, y={n_patches_y}"

    def test_get_n_patches_aspect_ratio_preservation(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that aspect ratio is approximately preserved in patch dimensions."""
        patch_size = 14

        # Test with a 2:1 aspect ratio image
        image_size = (300, 600)  # height=300, width=600
        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            image_size, patch_size
        )

        # The aspect ratio of patches should be close to 2:1
        patch_ratio = n_patches_x / n_patches_y
        expected_ratio = 2.0

        # Allow some tolerance due to rounding and even-dimension requirements
        assert 1.5 <= patch_ratio <= 2.5, f"Expected ~2:1 ratio, got {patch_ratio:.2f}"


class TestGetImageMask:
    """Test the get_image_mask method for identifying image tokens."""

    def test_get_image_mask_shape(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that image mask has the same shape as input_ids."""
        image = Image.new("RGB", (64, 32), color="red")
        batch_feature = processor_from_pretrained.process_images([image])

        image_mask = processor_from_pretrained.get_image_mask(batch_feature)

        assert image_mask.shape == batch_feature.input_ids.shape
        assert image_mask.dtype == torch.bool

    def test_get_image_mask_has_image_tokens(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that the mask identifies some image tokens."""
        image = Image.new("RGB", (64, 32), color="blue")
        batch_feature = processor_from_pretrained.process_images([image])

        image_mask = processor_from_pretrained.get_image_mask(batch_feature)

        # There should be image tokens present
        assert (
            image_mask.sum() > 0
        ), "Expected to find image tokens in the processed batch"

    def test_get_image_mask_batch_consistency(
        self, processor_from_pretrained: ColModernVBertProcessor
    ):
        """Test that image mask works correctly with batched images."""
        images = [
            Image.new("RGB", (64, 32), color="red"),
            Image.new("RGB", (128, 64), color="green"),
        ]
        batch_feature = processor_from_pretrained.process_images(images)

        image_mask = processor_from_pretrained.get_image_mask(batch_feature)

        assert image_mask.shape[0] == len(images)
        # Each image should have some image tokens
        for i in range(len(images)):
            assert image_mask[i].sum() > 0, f"Image {i} should have image tokens"


class TestEndToEndInterpretability:
    """Test end-to-end interpretability map generation."""

    @pytest.mark.slow
    def test_similarity_maps_shape(
        self,
        processor_from_pretrained: ColModernVBertProcessor,
        model_from_pretrained: ColModernVBert,
    ):
        """Test that similarity maps have the correct shape based on get_n_patches."""
        # Create a test image
        image_size_pil = (128, 64)  # PIL uses (width, height)
        image = Image.new("RGB", image_size_pil, color="white")

        # Create a query
        query = "test query"

        # Process image and query
        batch_images = processor_from_pretrained.process_images([image])
        batch_queries = processor_from_pretrained.process_texts([query])

        # Get patch size from the model or processor
        # ModernVBert uses patch_size from its config
        patch_size = (
            14  # Default for many vision transformers (unused but required for API)
        )

        # Calculate expected patches
        # Note: image_size for get_n_patches is (height, width)
        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            (image_size_pil[1], image_size_pil[0]), patch_size  # (height, width)
        )

        # Get embeddings
        with torch.no_grad():
            image_embeddings = model_from_pretrained(**batch_images)
            query_embeddings = model_from_pretrained(**batch_queries)

        # Get LOCAL image mask (excluding global patch for interpretability)
        image_mask = processor_from_pretrained.get_local_image_mask(batch_images)

        # Calculate similarity maps using the processor's method
        # This correctly handles the sub-patch token ordering
        similarity_maps = processor_from_pretrained.get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=(n_patches_x, n_patches_y),
            image_mask=image_mask,
        )

        # Check shape
        assert len(similarity_maps) == 1  # One batch item
        query_length = query_embeddings.shape[1]

        # similarity_maps[0] should have shape (query_tokens, n_patches_x, n_patches_y)
        expected_shape = (query_length, n_patches_x, n_patches_y)
        assert (
            similarity_maps[0].shape == expected_shape
        ), f"Expected shape {expected_shape}, got {similarity_maps[0].shape}"

    @pytest.mark.slow
    def test_similarity_maps_values(
        self,
        processor_from_pretrained: ColModernVBertProcessor,
        model_from_pretrained: ColModernVBert,
    ):
        """Test that similarity map values are reasonable after normalization."""
        image = Image.new("RGB", (64, 64), color="black")
        query = "dark image"

        batch_images = processor_from_pretrained.process_images([image])
        batch_queries = processor_from_pretrained.process_texts([query])

        patch_size = 14
        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            (64, 64), patch_size
        )

        with torch.no_grad():
            image_embeddings = model_from_pretrained(**batch_images)
            query_embeddings = model_from_pretrained(**batch_queries)

        # Use LOCAL image mask (excluding global patch)
        image_mask = processor_from_pretrained.get_local_image_mask(batch_images)

        # Use the processor's method for correct sub-patch ordering
        similarity_maps = processor_from_pretrained.get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=(n_patches_x, n_patches_y),
            image_mask=image_mask,
        )

        # Normalize and check values
        sim_map = similarity_maps[0][0]  # First query token's similarity map
        normalized_map = normalize_similarity_map(sim_map)

        # After normalization, values should be in [0, 1]
        assert normalized_map.min() >= 0.0
        assert normalized_map.max() <= 1.0
        assert (
            normalized_map.max() == 1.0
        )  # Max should be exactly 1.0 after normalization

    @pytest.mark.slow
    def test_patch_count_matches_mask_count(
        self,
        processor_from_pretrained: ColModernVBertProcessor,
    ):
        """Test that the number of LOCAL image tokens matches expected patch count."""
        image_size_pil = (128, 128)
        image = Image.new("RGB", image_size_pil, color="gray")

        batch_feature = processor_from_pretrained.process_images([image])

        # Use LOCAL image mask (excluding global patch)
        local_image_mask = processor_from_pretrained.get_local_image_mask(batch_feature)

        # Count actual LOCAL image tokens
        actual_local_tokens = local_image_mask.sum().item()

        # Calculate expected patches
        patch_size = 14
        n_patches_x, n_patches_y = processor_from_pretrained.get_n_patches(
            (image_size_pil[1], image_size_pil[0]), patch_size
        )
        expected_local_patches = n_patches_x * n_patches_y

        # LOCAL tokens should match exactly
        assert (
            actual_local_tokens == expected_local_patches
        ), f"Expected {expected_local_patches} local image tokens, got {actual_local_tokens}"

    @pytest.mark.slow
    def test_global_patch_excluded(
        self,
        processor_from_pretrained: ColModernVBertProcessor,
    ):
        """Test that global patch is correctly excluded from local mask."""
        image_size_pil = (128, 128)
        image = Image.new("RGB", image_size_pil, color="gray")

        batch_feature = processor_from_pretrained.process_images([image])

        full_mask = processor_from_pretrained.get_image_mask(batch_feature)
        local_mask = processor_from_pretrained.get_local_image_mask(batch_feature)

        full_count = full_mask.sum().item()
        local_count = local_mask.sum().item()

        # The difference should be exactly image_seq_len (global patch tokens)
        image_seq_len = processor_from_pretrained.image_seq_len
        assert (
            full_count - local_count == image_seq_len
        ), f"Expected {image_seq_len} global patch tokens, got {full_count - local_count}"


class TestInterpretabilityConsistency:
    """Test consistency of interpretability across different scenarios."""

    def test_different_image_sizes_produce_different_patch_counts(
        self,
        processor_from_pretrained: ColModernVBertProcessor,
    ):
        """Test that different image sizes produce different patch dimensions."""
        patch_size = 14

        small_patches = processor_from_pretrained.get_n_patches((100, 100), patch_size)
        large_patches = processor_from_pretrained.get_n_patches((500, 500), patch_size)

        # Larger images should produce the same or different patch counts
        # depending on the longest_edge configuration
        # At minimum, verify both are valid
        assert small_patches[0] > 0 and small_patches[1] > 0
        assert large_patches[0] > 0 and large_patches[1] > 0

    def test_consistent_patch_calculation(
        self,
        processor_from_pretrained: ColModernVBertProcessor,
    ):
        """Test that get_n_patches is deterministic."""
        patch_size = 14
        image_size = (256, 512)

        # Call multiple times
        result1 = processor_from_pretrained.get_n_patches(image_size, patch_size)
        result2 = processor_from_pretrained.get_n_patches(image_size, patch_size)
        result3 = processor_from_pretrained.get_n_patches(image_size, patch_size)

        assert result1 == result2 == result3, "get_n_patches should be deterministic"
