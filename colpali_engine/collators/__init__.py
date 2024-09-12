try:
    # NOTE: `HardNegCollator` requires the optional `datasets` dependency.
    from .hard_neg_collator import HardNegCollator
except ImportError:
    pass

from .visual_retriever_collator import VisualRetrieverCollator
