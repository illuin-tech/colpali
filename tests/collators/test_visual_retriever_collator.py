from typing import Generator, cast

import pytest
from PIL import Image

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor


class TestColPaliCollator:
    @pytest.fixture(scope="class")
    def colpali_processor_path(self) -> str:
        return "vidore/colpali-v1.2"

    @pytest.fixture(scope="class")
    def processor_from_pretrained(self, colpali_processor_path: str) -> Generator[ColPaliProcessor, None, None]:
        yield cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(colpali_processor_path))

    @pytest.fixture(scope="class")
    def colpali_collator(
        self, processor_from_pretrained: ColPaliProcessor
    ) -> Generator[VisualRetrieverCollator, None, None]:
        yield VisualRetrieverCollator(processor=processor_from_pretrained)

    def test_colpali_collator_call(self, colpali_collator: VisualRetrieverCollator):
        example_image = Image.new("RGB", (16, 16), color="red")
        examples = [
            {"query": "What is this?", "image": example_image},
        ]

        result = colpali_collator(examples)

        assert isinstance(result, dict)
        assert "doc_input_ids" in result
        assert "doc_attention_mask" in result
        assert "doc_pixel_values" in result
        assert "query_input_ids" in result
        assert "query_attention_mask" in result

    def test_colpali_collator_call_with_neg_images(self, colpali_collator: VisualRetrieverCollator):
        example_image = Image.new("RGB", (16, 16), color="red")
        neg_image = Image.new("RGB", (16, 16), color="blue")
        examples = [
            {
                "query": "What is this?",
                "image": example_image,
                "neg_image": neg_image,
            },
        ]

        result = colpali_collator(examples)

        assert isinstance(result, dict)
        assert "doc_input_ids" in result
        assert "doc_attention_mask" in result
        assert "doc_pixel_values" in result
        assert "query_input_ids" in result
        assert "query_attention_mask" in result
        assert "neg_doc_input_ids" in result
        assert "neg_doc_attention_mask" in result
        assert "neg_doc_pixel_values" in result
