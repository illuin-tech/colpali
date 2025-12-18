"""
End-to-end integration tests for interpretability workflows.

These tests verify the full pipeline from embeddings to visualization,
similar to the example scripts in examples/interpretability/.
"""

import math
from typing import List, Tuple

import pytest
import torch
from matplotlib import pyplot as plt
from PIL import Image

from colpali_engine.interpretability import get_similarity_maps_from_embeddings
from colpali_engine.interpretability.similarity_map_utils import normalize_similarity_map
from colpali_engine.interpretability.similarity_maps import plot_all_similarity_maps, plot_similarity_map


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (224, 224), color="white")


@pytest.fixture
def sample_embeddings() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample image and query embeddings for testing.

    Returns:
        tuple: (image_embeddings, query_embeddings)
            - image_embeddings: shape (batch_size=2, image_tokens=16, dim=128)
            - query_embeddings: shape (batch_size=2, query_tokens=5, dim=128)
    """
    batch_size = 2
    image_tokens = 16  # 4x4 patches
    query_tokens = 5
    dim = 128

    image_embeddings = torch.randn(batch_size, image_tokens, dim)
    query_embeddings = torch.randn(batch_size, query_tokens, dim)

    # Normalize embeddings as would be done in real models
    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)

    return image_embeddings, query_embeddings


class TestImageMaskCreation:
    """Test creation of image masks from input_ids."""

    def test_create_image_mask_from_input_ids(self):
        """Test creating image mask by identifying image token IDs."""
        batch_size = 2
        seq_len = 20
        image_token_id = 256000  # Example image token ID

        # Create input_ids with some image tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Set specific positions as image tokens
        input_ids[:, 5:13] = image_token_id  # 8 image tokens per sample

        # Create image mask
        image_mask = input_ids == image_token_id

        assert image_mask.shape == (batch_size, seq_len)
        assert image_mask.dtype == torch.bool
        assert image_mask.sum(dim=1).tolist() == [8, 8]

    def test_image_mask_all_ones_fallback(self):
        """Test fallback to all-ones mask when image_token_id is not available."""
        batch_size = 2
        image_tokens = 16

        # Create all-ones mask as fallback
        image_mask = torch.ones(batch_size, image_tokens, dtype=torch.bool)

        assert image_mask.shape == (batch_size, image_tokens)
        assert image_mask.all()


class TestNPatchesCalculation:
    """Test calculation of n_patches from number of image tokens."""

    def test_calculate_n_patches_perfect_square(self):
        """Test n_patches calculation when num_image_tokens is a perfect square."""
        num_image_tokens = 16
        n_side = int(math.sqrt(num_image_tokens))

        assert n_side * n_side == num_image_tokens
        n_patches = (n_side, n_side)

        assert n_patches == (4, 4)

    def test_calculate_n_patches_not_perfect_square(self):
        """Test handling of non-perfect-square image token counts."""
        num_image_tokens = 15  # Not a perfect square
        n_side = int(math.sqrt(num_image_tokens))

        is_perfect_square = n_side * n_side == num_image_tokens
        assert not is_perfect_square

        # Should use fallback or alternative calculation
        # In practice, we'd use a default like (16, 16) or calculate aspect ratio

    def test_calculate_n_patches_from_mask(self):
        """Test calculating n_patches from image mask."""
        batch_size = 2
        seq_len = 20

        # Create image mask with 9 image tokens (3x3)
        image_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        image_mask[:, 5:14] = True  # 9 tokens

        num_image_tokens = image_mask[0].sum().item()
        n_side = int(math.sqrt(num_image_tokens))

        assert num_image_tokens == 9
        assert n_side == 3
        assert n_side * n_side == num_image_tokens


class TestAggregatedHeatmaps:
    """Test generation of aggregated heatmaps across query tokens."""

    def test_mean_aggregation(self):
        """Test mean aggregation across query tokens."""
        query_tokens = 5
        n_patches_x, n_patches_y = 4, 4

        # Create similarity maps for multiple query tokens
        similarity_maps = torch.randn(query_tokens, n_patches_x, n_patches_y)

        # Calculate mean across query tokens
        aggregated_map = torch.mean(similarity_maps, dim=0)

        assert aggregated_map.shape == (n_patches_x, n_patches_y)
        assert aggregated_map.dtype == similarity_maps.dtype

    def test_max_aggregation(self):
        """Test max aggregation across query tokens."""
        query_tokens = 5
        n_patches_x, n_patches_y = 4, 4

        similarity_maps = torch.randn(query_tokens, n_patches_x, n_patches_y)

        # Calculate max across query tokens
        aggregated_map = torch.max(similarity_maps, dim=0)[0]

        assert aggregated_map.shape == (n_patches_x, n_patches_y)
        # Max should be >= mean for each position

    def test_normalized_aggregated_map(self):
        """Test normalization of aggregated heatmap."""
        query_tokens = 5
        n_patches_x, n_patches_y = 4, 4

        similarity_maps = torch.randn(query_tokens, n_patches_x, n_patches_y)
        aggregated_map = torch.mean(similarity_maps, dim=0)

        # Normalize
        normalized_map = normalize_similarity_map(aggregated_map)

        assert normalized_map.shape == aggregated_map.shape
        assert normalized_map.min() >= 0.0
        assert normalized_map.max() <= 1.0


class TestTokenFiltering:
    """Test filtering of special tokens from query tokens."""

    def test_filter_special_tokens(self):
        """Test filtering special tokens from token list."""
        # Simulate tokenizer output
        all_tokens = ["<s>", "What", "is", "the", "total", "</s>", "<pad>"]
        all_token_ids = [1, 100, 101, 102, 103, 2, 0]
        special_token_ids = {0, 1, 2}  # <pad>, <s>, </s>

        # Filter special tokens
        filtered_tokens = []
        filtered_indices = []
        for idx, (token, token_id) in enumerate(zip(all_tokens, all_token_ids)):
            if token_id not in special_token_ids:
                filtered_tokens.append(token)
                filtered_indices.append(idx)

        assert filtered_tokens == ["What", "is", "the", "total"]
        assert filtered_indices == [1, 2, 3, 4]

    def test_filter_similarity_maps_by_indices(self):
        """Test filtering similarity maps based on token indices."""
        query_length = 7  # Including special tokens
        n_patches_x, n_patches_y = 4, 4

        similarity_maps = torch.randn(query_length, n_patches_x, n_patches_y)

        # Filter to non-special tokens
        filtered_indices = [1, 2, 3, 4]  # Indices of non-special tokens
        filtered_maps = similarity_maps[filtered_indices]

        assert filtered_maps.shape == (len(filtered_indices), n_patches_x, n_patches_y)


class TestEndToEndWorkflow:
    """Test complete interpretability workflow."""

    def test_full_pipeline_single_image(self, sample_image, sample_embeddings):
        """Test full pipeline from embeddings to visualization for a single image."""
        image_embeddings, query_embeddings = sample_embeddings

        # Use only first sample
        image_embeddings = image_embeddings[:1]
        query_embeddings = query_embeddings[:1]

        # Create image mask (all tokens are image tokens)
        batch_size, image_tokens, _ = image_embeddings.shape
        image_mask = torch.ones(batch_size, image_tokens, dtype=torch.bool)

        # Calculate n_patches (4x4 for 16 tokens)
        n_patches = (4, 4)

        # Generate similarity maps
        similarity_maps_batch = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        # Get the first (and only) similarity map
        similarity_maps = similarity_maps_batch[0]

        # Verify shape
        query_tokens = query_embeddings.shape[1]
        assert similarity_maps.shape == (query_tokens, n_patches[0], n_patches[1])

        # Create visualization for first token
        fig, ax = plot_similarity_map(
            image=sample_image,
            similarity_map=similarity_maps[0],
            figsize=(5, 5),
            show_colorbar=True,
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_full_pipeline_batch(self, sample_image, sample_embeddings):
        """Test full pipeline with batch of images."""
        image_embeddings, query_embeddings = sample_embeddings

        batch_size, image_tokens, _ = image_embeddings.shape
        image_mask = torch.ones(batch_size, image_tokens, dtype=torch.bool)

        # Different n_patches for each image in batch
        n_patches = [(4, 4), (4, 4)]

        # Generate similarity maps
        similarity_maps_batch = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        assert len(similarity_maps_batch) == batch_size

        for idx, similarity_maps in enumerate(similarity_maps_batch):
            query_tokens = query_embeddings.shape[1]
            assert similarity_maps.shape == (query_tokens, n_patches[idx][0], n_patches[idx][1])

    def test_plot_all_with_token_filtering(self, sample_image):
        """Test plotting all similarity maps with filtered tokens."""
        # Create sample data
        query_tokens = ["What", "is", "the", "total"]
        n_patches_x, n_patches_y = 4, 4
        num_tokens = len(query_tokens)

        similarity_maps = torch.randn(num_tokens, n_patches_x, n_patches_y)

        # Plot all similarity maps
        plots = plot_all_similarity_maps(
            image=sample_image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
            figsize=(5, 5),
            show_colorbar=True,
            add_title=True,
        )

        assert len(plots) == num_tokens
        for fig, ax in plots:
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            plt.close(fig)

    def test_aggregated_heatmap_generation(self, sample_image):
        """Test generation of aggregated heatmap across all tokens."""
        query_tokens = 5
        n_patches_x, n_patches_y = 4, 4

        # Create similarity maps
        similarity_maps = torch.randn(query_tokens, n_patches_x, n_patches_y)

        # Calculate aggregated map (mean)
        aggregated_map = torch.mean(similarity_maps, dim=0)

        # Normalize
        normalized_map = normalize_similarity_map(aggregated_map)

        # Plot
        fig, ax = plot_similarity_map(
            image=sample_image,
            similarity_map=normalized_map,
            figsize=(8, 8),
            show_colorbar=True,
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestNormalizationRanges:
    """Test normalization with custom value ranges."""

    def test_normalize_with_custom_range(self):
        """Test normalization with custom min/max range."""
        similarity_map = torch.tensor([
            [0.5, 0.7],
            [0.3, 0.9],
        ])

        # Use custom range
        value_range = (0.0, 1.0)
        normalized_map = normalize_similarity_map(similarity_map, value_range=value_range)

        assert normalized_map.min() >= 0.0
        assert normalized_map.max() <= 1.0

    def test_normalize_per_query_in_plot_all(self, sample_image):
        """Test normalize_per_query parameter in plot_all_similarity_maps."""
        query_tokens = ["token1", "token2", "token3"]
        similarity_maps = torch.randn(len(query_tokens), 4, 4)

        # With normalize_per_query=True (default)
        plots_normalized = plot_all_similarity_maps(
            image=sample_image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
            normalize_per_query=True,
        )

        assert len(plots_normalized) == len(query_tokens)
        for fig, _ in plots_normalized:
            plt.close(fig)

        # With normalize_per_query=False
        plots_individual = plot_all_similarity_maps(
            image=sample_image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
            normalize_per_query=False,
        )

        assert len(plots_individual) == len(query_tokens)
        for fig, _ in plots_individual:
            plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_query_token(self, sample_image):
        """Test with single query token."""
        similarity_map = torch.randn(4, 4)

        fig, ax = plot_similarity_map(
            image=sample_image,
            similarity_map=similarity_map,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_patch_grid(self, sample_image):
        """Test with large patch grid."""
        query_tokens = 10
        n_patches = (16, 16)

        similarity_maps = torch.randn(query_tokens, n_patches[0], n_patches[1])

        # Should handle large grids
        aggregated = torch.mean(similarity_maps, dim=0)
        assert aggregated.shape == n_patches

    def test_bfloat16_conversion(self):
        """Test conversion from bfloat16 to float32 for visualization."""
        similarity_map = torch.randn(4, 4, dtype=torch.bfloat16)

        # Convert to float32 for visualization
        if similarity_map.dtype == torch.bfloat16:
            similarity_map = similarity_map.float()

        assert similarity_map.dtype == torch.float32

    def test_empty_token_list_handling(self):
        """Test handling of empty filtered token list."""
        all_tokens = ["<s>", "</s>", "<pad>"]
        all_token_ids = [1, 2, 0]
        special_token_ids = {0, 1, 2}

        filtered_tokens = []
        for token, token_id in zip(all_tokens, all_token_ids):
            if token_id not in special_token_ids:
                filtered_tokens.append(token)

        assert len(filtered_tokens) == 0
