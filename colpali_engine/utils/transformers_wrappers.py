import importlib

if importlib.util.find_spec("transformers") is not None:
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer

    class AllPurposeWrapper:
        def __new__(cls, class_to_instanciate, *args, **kwargs):
            return class_to_instanciate.from_pretrained(*args, **kwargs)

    class AutoProcessorWrapper:
        def __new__(cls, *args, **kwargs):
            return AutoProcessor.from_pretrained(*args, **kwargs)

    class AutoTokenizerWrapper(PreTrainedTokenizer):
        def __new__(cls, *args, **kwargs):
            return AutoTokenizer.from_pretrained(*args, **kwargs)

else:
    raise ModuleNotFoundError("Transformers must be loaded")
