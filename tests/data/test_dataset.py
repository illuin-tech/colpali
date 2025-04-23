from typing import Generator

import pytest
from PIL import Image

from colpali_engine.data.corpus import SimpleCorpus
from colpali_engine.data.dataset import IRColumn, ColpaliEngineDataset


class TestColPaliCollator:
    @pytest.fixture(scope="class")
    def colpali_corpus(self) -> Generator[SimpleCorpus, None, None]:
        # Mock data for the corpus
        corpus_data = [
            {"doc": Image.new("RGB", (16, 16), color="red")},
            {"doc": Image.new("RGB", (16, 16), color="blue")},
        ]
        yield SimpleCorpus(corpus_data=corpus_data, id_column=None)

    @pytest.fixture(scope="class")
    def colpali_dataset(self) -> Generator[ColpaliEngineDataset, None, None]:
        # Mock data for the dataset
        data = [
            {"query": "What is this?", "image": Image.new("RGB", (16, 16), color="red")},
        ]
        yield ColpaliEngineDataset(
            data=data,
            query_column="query",
            pos_target_column="image",
        )

    @pytest.fixture(scope="class")
    def colpali_dataset_with_negs(self) -> Generator[ColpaliEngineDataset, None, None]:
        # Mock data for the dataset
        data = [
            {
                "query": "What is this?",
                "image": Image.new("RGB", (16, 16), color="red"),
                "neg_image": Image.new("RGB", (16, 16), color="blue"),
            },
        ]
        yield ColpaliEngineDataset(
            data=data,
            query_column="query",
            pos_target_column="image",
            neg_target_column="neg_image",
        )

    @pytest.fixture(scope="class")
    def colpali_dataset_with_corpus(self, colpali_corpus: ColpaliEngineDataset) -> Generator[ColpaliEngineDataset, None, None]:
        data = [
            {"query": "What is this?", "image": Image.new("RGB", (16, 16), color="red")},
        ]
        yield ColpaliEngineDataset(
            data=data,
            query_column="query",
            pos_target_column=IRColumn("image", corpus_column="doc"),
            corpus=colpali_corpus,
        )

    def test_colpali_dataset_call(self, colpali_dataset: ColpaliEngineDataset):
        result = colpali_dataset[0]
        assert isinstance(result, dict)
        assert ColpaliEngineDataset.QUERY_KEY in result
        assert ColpaliEngineDataset.POS_TARGET_KEY in result
        assert ColpaliEngineDataset.NEG_TARGET_KEY in result and result[ColpaliEngineDataset.NEG_TARGET_KEY] is None

    def test_colpali_dataset_call_with_corpus(self, colpali_dataset_with_corpus: ColpaliEngineDataset):
        result = colpali_dataset_with_corpus[0]
        assert isinstance(result, dict)
        assert ColpaliEngineDataset.QUERY_KEY in result
        assert ColpaliEngineDataset.POS_TARGET_KEY in result
        assert ColpaliEngineDataset.NEG_TARGET_KEY in result and result[ColpaliEngineDataset.NEG_TARGET_KEY] is None

    def test_colpali_dataset_call_with_neg_images(self, colpali_dataset_with_negs: ColpaliEngineDataset):
        result = colpali_dataset_with_negs[0]
        assert isinstance(result, dict)
        assert ColpaliEngineDataset.QUERY_KEY in result
        assert ColpaliEngineDataset.POS_TARGET_KEY in result
        assert ColpaliEngineDataset.NEG_TARGET_KEY in result
