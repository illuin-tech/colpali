import pytest
import torch

from colpali_engine.interpretability.similarity_map_utils import get_similarity_maps_from_embeddings
from colpali_engine.interpretability.similarity_maps import normalize_similarity_map


class TestNormalizeSimilarityMap:
    def test_normalize_similarity_map_2d_ones(self):
        similarity_map = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
        normalized_map = normalize_similarity_map(similarity_map)
        expected_map = torch.zeros_like(similarity_map)
        assert torch.allclose(normalized_map, expected_map, atol=1e-6)

    def test_normalize_similarity_map_2d(self):
        similarity_map = torch.tensor(
            [
                [1.0, 1.0],
                [0.0, -1.0],
            ]
        )
        normalized_map = normalize_similarity_map(similarity_map)
        expected_map = torch.tensor(
            [
                [1.0, 1.0],
                [0.5, 0.0],
            ]
        )
        assert torch.allclose(normalized_map, expected_map, atol=1e-6)

    def test_normalize_similarity_map_3d_ones(self):
        similarity_map = torch.tensor(
            [
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                [
                    [2.0, 2.0],
                    [2.0, 2.0],
                ],
            ]
        )
        normalized_map = normalize_similarity_map(similarity_map)
        expected_map = torch.zeros_like(similarity_map)

        assert torch.allclose(normalized_map, expected_map, atol=1e-6)


class TestGetSimilarityMapsFromEmbeddings:
    def test_get_similarity_maps_from_embeddings(self):
        # Define test parameters
        batch_size = 2
        image_tokens = 6  # Total number of image tokens
        query_tokens = 3
        dim = 4  # Embedding dimension

        # Create dummy image embeddings and query embeddings
        image_embeddings = torch.randn(batch_size, image_tokens, dim)
        query_embeddings = torch.randn(batch_size, query_tokens, dim)

        # Define n_patches as a tuple (h, w), ensuring h * w equals image_tokens
        n_patches = (2, 3)  # For instance, 2 rows and 3 columns

        # Create an optional image attention mask (all ones, no padding)
        image_mask = torch.ones(batch_size, image_tokens, dtype=torch.bool)

        # Call the function under test
        similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        # Assertions to validate the output
        assert isinstance(similarity_maps, list), "Output should be a list of tensors."
        assert len(similarity_maps) == batch_size, "Output list length should match batch size."

        for idx, similarity_map in enumerate(similarity_maps):
            expected_shape = (query_tokens, n_patches[0], n_patches[1])
            assert similarity_map.shape == expected_shape, (
                f"Similarity map at index {idx} has shape {similarity_map.shape}, " f"expected {expected_shape}."
            )

    def test_get_similarity_maps_with_varied_n_patches(self):
        # Define test parameters
        batch_size = 2
        image_tokens_list = [6, 8]  # Different number of tokens for each image
        query_tokens = 3
        dim = 4  # Embedding dimension

        # Create dummy image embeddings with padding to match the maximum tokens
        max_image_tokens = max(image_tokens_list)
        image_embeddings = torch.randn(batch_size, max_image_tokens, dim)
        query_embeddings = torch.randn(batch_size, query_tokens, dim)

        # Define n_patches as a list of tuples
        n_patches = [(2, 3), (2, 4)]  # Different for each image

        # Create image attention masks for variable image tokens
        image_mask = torch.zeros(batch_size, max_image_tokens, dtype=torch.bool)
        for idx, tokens in enumerate(image_tokens_list):
            image_mask[idx, :tokens] = 1

        # Call the function under test
        similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        # Assertions to validate the output
        assert isinstance(similarity_maps, list), "Output should be a list of tensors."
        assert len(similarity_maps) == batch_size, "Output list length should match batch size."

        for idx, similarity_map in enumerate(similarity_maps):
            expected_shape = (query_tokens, n_patches[idx][0], n_patches[idx][1])
            assert similarity_map.shape == expected_shape, (
                f"Similarity map at index {idx} has shape {similarity_map.shape}, " f"expected {expected_shape}."
            )

    def test_get_similarity_maps_with_incorrect_n_patches(self):
        # Define test parameters
        batch_size = 1
        image_tokens = 6  # Total number of image tokens
        query_tokens = 2
        dim = 5  # Embedding dimension

        # Create dummy image embeddings and query embeddings
        image_embeddings = torch.randn(batch_size, image_tokens, dim)
        query_embeddings = torch.randn(batch_size, query_tokens, dim)

        # Define incorrect n_patches that do not match image_tokens
        n_patches = (2, 2)  # 2*2 != 6

        # Create image attention masks for variable image tokens
        image_mask = torch.ones(batch_size, image_tokens, dtype=torch.bool)

        # Expect an error due to shape mismatch
        with pytest.raises(ValueError):
            get_similarity_maps_from_embeddings(
                image_embeddings=image_embeddings,
                query_embeddings=query_embeddings,
                n_patches=n_patches,
                image_mask=image_mask,
            )

    def test_get_similarity_maps_with_padding(self):
        # Define test parameters
        batch_size = 1
        image_tokens = 9  # Total number of image tokens
        query_tokens = 2
        dim = 5  # Embedding dimension

        # Create dummy image embeddings and query embeddings
        image_embeddings = torch.randn(batch_size, image_tokens, dim)
        query_embeddings = torch.randn(batch_size, query_tokens, dim)

        # Define n_patches as a tuple
        n_patches = (3, 2)

        # Create an image attention mask with padding
        image_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)

        # Call the function under test
        similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        # Assertions to validate the output
        assert isinstance(similarity_maps, list), "Output should be a list of tensors."
        assert len(similarity_maps) == batch_size, "Output list length should match batch size."

        similarity_map = similarity_maps[0]
        expected_shape = (query_tokens, n_patches[0], n_patches[1])
        assert (
            similarity_map.shape == expected_shape
        ), f"Similarity map has shape {similarity_map.shape}, expected {expected_shape}."
