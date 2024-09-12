import importlib.util

if importlib.util.find_spec("datasets"):
    from .hard_neg_collator import HardNegCollator

from .visual_retriever_collator import VisualRetrieverCollator
