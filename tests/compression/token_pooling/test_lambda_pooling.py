import torch

from colpali_engine.compression.token_pooling import LambdaTokenPooler, TokenPoolingOutput

from .example_pooling_functions import (
    halve_pool_func,
    identity_pool_func,
    mean_pool_func,
)


def test_lambda_token_pooler_initialization():
    pooler = LambdaTokenPooler(pool_func=identity_pool_func)
    assert pooler.pool_func == identity_pool_func


def test_lambda_token_pooler_with_identity_function(sample_embedding: torch.Tensor):
    pooler = LambdaTokenPooler(pool_func=identity_pool_func)
    outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    assert torch.allclose(outputs.pooled_embeddings[0], sample_embedding)


def test_lambda_token_pooler_with_mean_pooling(sample_embedding: torch.Tensor):
    pooler = LambdaTokenPooler(pool_func=mean_pool_func)
    outputs = pooler.pool_embeddings([sample_embedding], return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    assert outputs.pooled_embeddings[0].shape[0] == 1
    assert outputs.pooled_embeddings[0].shape[1] == sample_embedding.shape[1]


def test_lambda_token_pooler_with_batched_input():
    batch_embeddings = torch.rand(3, 10, 768)

    pooler = LambdaTokenPooler(pool_func=halve_pool_func)
    outputs = pooler.pool_embeddings(batch_embeddings, return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    assert isinstance(outputs.pooled_embeddings, torch.Tensor)
    assert outputs.pooled_embeddings.shape == (3, 5, 768)


def test_lambda_token_pooler_with_list_input():
    list_embeddings = [
        torch.rand(10, 768),
        torch.rand(15, 768),
    ]

    pooler = LambdaTokenPooler(pool_func=halve_pool_func)
    outputs = pooler.pool_embeddings(list_embeddings, return_dict=True)

    assert isinstance(outputs, TokenPoolingOutput)
    assert len(outputs.pooled_embeddings) == len(list_embeddings)

    # First embedding should be pooled to half size
    assert outputs.pooled_embeddings[0].shape[0] == 5
    assert outputs.pooled_embeddings[0].shape[1] == 768

    # Second embedding should be pooled to half size (rounded up)
    assert outputs.pooled_embeddings[1].shape[0] == 8
    assert outputs.pooled_embeddings[1].shape[1] == 768
