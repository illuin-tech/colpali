from typing import Generator

import pytest
from PIL import Image

from colpali_engine.data.corpus import SimpleCorpus
from colpali_engine.data.dataset import IRColumn, ColPaliEngineDataset


class TestColPaliEngineDataset:
    @pytest.fixture(scope="class")
    def corpus(self) -> Generator[SimpleCorpus, None, None]:
        # Mock data for the corpus
        corpus_data = [
            {"doc": Image.new("RGB", (16, 16), color="red")},
            {"doc": Image.new("RGB", (16, 16), color="blue")},
        ]
        yield SimpleCorpus(corpus_data=corpus_data, id_column=None)

    @pytest.fixture(scope="class")
    def ir_dataset(self) -> Generator[ColPaliEngineDataset, None, None]:
        # Mock data for the dataset
        data = [
            {"query": "What is this?", "image": Image.new("RGB", (16, 16), color="red")},
        ]
        yield ColPaliEngineDataset(
            data=data,
            query_column="query",
            pos_target_column="image",
        )

    @pytest.fixture(scope="class")
    def ir_dataset_with_negs(self) -> Generator[ColPaliEngineDataset, None, None]:
        # Mock data for the dataset
        data = [
            {
                "query": "What is this?",
                "image": Image.new("RGB", (16, 16), color="red"),
                "neg_image": Image.new("RGB", (16, 16), color="blue"),
            },
        ]
        yield ColPaliEngineDataset(
            data=data,
            query_column="query",
            pos_target_column="image",
            neg_target_column="neg_image",
        )

    @pytest.fixture(scope="class")
    def ir_dataset_with_corpus(self, corpus: ColPaliEngineDataset) -> Generator[ColPaliEngineDataset, None, None]:
        data = [
            {"query": "What is this?", "image": Image.new("RGB", (16, 16), color="red")},
        ]
        yield ColPaliEngineDataset(
            data=data,
            query_column="query",
            pos_target_column=IRColumn("image", corpus_column="doc"),
            corpus=corpus,
        )

    def test_ir_dataset_call(self, ir_dataset: ColPaliEngineDataset):
        result = ir_dataset[0]
        assert isinstance(result, dict)
        assert ColPaliEngineDataset.QUERY_KEY in result
        assert ColPaliEngineDataset.POS_TARGET_KEY in result
        assert ColPaliEngineDataset.NEG_TARGET_KEY in result and result[ColPaliEngineDataset.NEG_TARGET_KEY] is None

    def test_ir_dataset_call_with_corpus(self, ir_dataset_with_corpus: ColPaliEngineDataset):
        result = ir_dataset_with_corpus[0]
        assert isinstance(result, dict)
        assert ColPaliEngineDataset.QUERY_KEY in result
        assert ColPaliEngineDataset.POS_TARGET_KEY in result
        assert ColPaliEngineDataset.NEG_TARGET_KEY in result and result[ColPaliEngineDataset.NEG_TARGET_KEY] is None

    def test_ir_dataset_call_with_neg_images(self, ir_dataset_with_negs: ColPaliEngineDataset):
        result = ir_dataset_with_negs[0]
        assert isinstance(result, dict)
        assert ColPaliEngineDataset.QUERY_KEY in result
        assert ColPaliEngineDataset.POS_TARGET_KEY in result
        assert ColPaliEngineDataset.NEG_TARGET_KEY in result
