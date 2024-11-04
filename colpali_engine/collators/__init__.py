import importlib.util

if importlib.util.find_spec("datasets"):
    from .corpus_query_collator import CorpusQueryCollator

from .visual_retriever_collator import VisualRetrieverCollator
