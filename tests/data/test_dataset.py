from typing import Generator

import pytest
from datasets import Dataset as HFDataset
from PIL import Image

from colpali_engine.data.dataset import ColPaliEngineDataset, Corpus


class TestColPaliEngineDataset:
    @pytest.fixture(scope="class")
    def corpus(self) -> Generator[Corpus, None, None]:
        # Mock data for the corpus
        corpus_data = HFDataset.from_list(
            [
                {"doc": Image.new("RGB", (16, 16), color="red")},
                {"doc": Image.new("RGB", (16, 16), color="blue")},
            ]
        )
        yield Corpus(corpus_data=corpus_data)

    @pytest.fixture(scope="class")
    def colpali_engine_dataset(self) -> Generator[ColPaliEngineDataset, None, None]:
        # Mock data for the dataset
        data = HFDataset.from_list(
            [
                {
                    ColPaliEngineDataset.QUERY_KEY: "What is this?",
                    ColPaliEngineDataset.POS_TARGET_KEY: Image.new("RGB", (16, 16), color="red"),
                },
            ]
        )
        yield ColPaliEngineDataset(
            data=data,
        )

    @pytest.fixture(scope="class")
    def colpali_engine_with_negs(self) -> Generator[ColPaliEngineDataset, None, None]:
        # Mock data for the dataset
        data = HFDataset.from_list(
            [
                {
                    "query": "What is this?",
                    "pos_target": Image.new("RGB", (16, 16), color="red"),
                    "neg_target": Image.new("RGB", (16, 16), color="blue"),
                },
            ]
        )
        yield ColPaliEngineDataset(
            data=data,
        )

    @pytest.fixture(scope="class")
    def colpali_engine_with_corpus(self, corpus: ColPaliEngineDataset) -> Generator[ColPaliEngineDataset, None, None]:
        data = HFDataset.from_list(
            [
                {"query": "What is this?", "pos_target": [0]},
            ]
        )
        yield ColPaliEngineDataset(
            data=data,
            external_document_corpus=corpus,
        )

    def test_colpali_engine_call(self, colpali_engine_dataset: ColPaliEngineDataset):
        result = colpali_engine_dataset[0]
        assert isinstance(result, dict)
        assert ColPaliEngineDataset.QUERY_KEY in result
        assert ColPaliEngineDataset.POS_TARGET_KEY in result
        assert ColPaliEngineDataset.NEG_TARGET_KEY in result and result[ColPaliEngineDataset.NEG_TARGET_KEY] is None

    def test_colpali_engine_call_with_corpus(self, colpali_engine_with_corpus: ColPaliEngineDataset):
        result = colpali_engine_with_corpus[0]
        assert isinstance(result, dict)
        assert ColPaliEngineDataset.QUERY_KEY in result
        assert ColPaliEngineDataset.POS_TARGET_KEY in result
        assert ColPaliEngineDataset.NEG_TARGET_KEY in result and result[ColPaliEngineDataset.NEG_TARGET_KEY] is None

    def test_colpali_engine_call_with_neg_images(self, colpali_engine_with_negs: ColPaliEngineDataset):
        result = colpali_engine_with_negs[0]
        assert isinstance(result, dict)
        assert ColPaliEngineDataset.QUERY_KEY in result
        assert ColPaliEngineDataset.POS_TARGET_KEY in result
        assert ColPaliEngineDataset.NEG_TARGET_KEY in result
