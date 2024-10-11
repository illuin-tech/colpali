from typing import List

import pytest
import torch
from matplotlib import pyplot as plt
from PIL import Image

from colpali_engine.interpretability.similarity_maps import plot_all_similarity_maps, plot_similarity_map


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def sample_similarity_maps() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
                [
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],
            ]
        ]
    )  # (1, 2, 2, 2)


@pytest.fixture
def sample_query_tokens() -> List[str]:
    return ["token1", "token2"]


def test_plot_similarity_map(sample_image, sample_similarity_maps):
    similarity_map = sample_similarity_maps[0, 0, :, :]
    fig, ax = plot_similarity_map(
        image=sample_image,
        similarity_map=similarity_map,
        resolution=(50, 50),
        figsize=(5, 5),
        show_colorbar=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_all_similarity_maps(sample_image, sample_similarity_maps, sample_query_tokens):
    plots = plot_all_similarity_maps(
        image=sample_image,
        query_tokens=sample_query_tokens,
        similarity_maps=sample_similarity_maps,
        resolution=(50, 50),
        figsize=(5, 5),
        show_colorbar=False,
    )
    assert len(plots) == len(sample_query_tokens)
    for fig, ax in plots:
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
