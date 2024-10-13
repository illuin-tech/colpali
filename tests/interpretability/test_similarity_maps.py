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
def sample_query_tokens() -> List[str]:
    return ["token1", "token2"]


def test_plot_similarity_map(sample_image):
    similarity_map = torch.rand(32, 32)
    fig, ax = plot_similarity_map(
        image=sample_image,
        similarity_map=similarity_map,
        figsize=(5, 5),
        show_colorbar=True,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_all_similarity_maps(sample_image, sample_query_tokens):
    similarity_maps = torch.rand(len(sample_query_tokens), 32, 32)

    plots = plot_all_similarity_maps(
        image=sample_image,
        query_tokens=sample_query_tokens,
        similarity_maps=similarity_maps,
        figsize=(5, 5),
        show_colorbar=False,
        add_title=True,
    )

    assert len(plots) == len(sample_query_tokens)
    for fig, ax in plots:
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
