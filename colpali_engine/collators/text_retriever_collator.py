from typing import List

from transformers import PreTrainedTokenizer


class TextRetrieverCollator:
    """
    Collator for training text-only retrievers.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        add_suffix: bool = True,
    ):
        self.tokenizer = tokenizer
        self.image_token_id = None
        self.max_length = max_length
        self.suffix = ""

        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if add_suffix:
            self.suffix = self.tokenizer.pad_token * 10

    def __call__(self, examples):
        """
        Collate function for text-only examples.
        """
        # Placeholders
        texts_doc: List[str] = []
        texts_query: List[str] = []

        # Process each example
        for example in examples:
            text_query = example["query"] + self.suffix
            text_doc = example["doc"]

            texts_doc.append(text_doc.strip())
            texts_query.append(text_query.strip())

        if self.tokenizer is None:
            raise ValueError("Tokenizer should be provided for text collator.")

        # Process the documents
        batch_doc = self.tokenizer(
            texts_doc,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        # Process the queries
        batch_query = self.tokenizer(
            texts_query,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        # Prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_all = {f"doc_{k}": v for k, v in batch_doc.items()}
        del batch_doc
        batch_query = {f"query_{k}": v for k, v in batch_query.items()}
        batch_all.update(batch_query)

        return batch_all
