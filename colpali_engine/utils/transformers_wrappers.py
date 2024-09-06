import importlib

from importlib import util

if util.find_spec("transformers") is not None:
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer

    class AllPurposeWrapper:
        def __new__(cls, class_to_instanciate, trust_remote_code=True, *args, **kwargs):
            return class_to_instanciate.from_pretrained(
                trust_remote_code=trust_remote_code,*args, **kwargs
            )

    class AutoProcessorWrapper:
        def __new__(cls, *args, **kwargs):
            return AutoProcessor.from_pretrained(trust_remote_code=True, min_pixels=256*28*28, max_pixels = 1280*28*28, *args, **kwargs)

    class AutoTokenizerWrapper(PreTrainedTokenizer):
        def __new__(cls, *args, **kwargs):
            return AutoTokenizer.from_pretrained(trust_remote_code=True, *args, **kwargs)

else:
    raise ModuleNotFoundError("Transformers must be loaded")
