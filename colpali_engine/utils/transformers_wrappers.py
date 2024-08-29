import importlib

from colpali_engine.models.bi_encoders import BiIdefics, BiNewSiglip, BiPaliMean, BiSigLIP
from colpali_engine.models.late_interaction import ColIdefics, ColNewSiglip, ColPali, ColSigLIP

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

    class AutoColModelWrapper:
        def __new__(cls, *args, **kwargs):
            pretrained_model_name_or_path = None
            if args:
                pretrained_model_name_or_path = args[0]
            elif kwargs:
                pretrained_model_name_or_path = kwargs["pretrained_model_name_or_path"]

            training_objective = kwargs.pop("training_objective", "colbertv1")

            if "idefics2" in pretrained_model_name_or_path:
                if training_objective == "biencoder":
                    return BiIdefics.from_pretrained(*args, **kwargs)
                return ColIdefics.from_pretrained(*args, **kwargs)
            elif "siglip" in pretrained_model_name_or_path:
                if training_objective == "biencoder_mean":
                    return BiSigLIP.from_pretrained(*args, **kwargs)
                elif training_objective == "colbertv1":
                    return ColSigLIP.from_pretrained(*args, **kwargs)
                else:
                    raise ValueError(f"Training objective {training_objective} not recognized")
            elif "paligemma" in pretrained_model_name_or_path:
                if training_objective == "biencoder_mean":
                    return BiPaliMean.from_pretrained(*args, **kwargs)
                elif training_objective == "biencoder_last":
                    return BiPaliLast.from_pretrained(*args, **kwargs)
                elif training_objective == "biencoder_mean_vision":
                    return BiNewSiglip.from_pretrained(*args, **kwargs)
                elif training_objective == "colbertv1_vision":
                    return ColNewSiglip.from_pretrained(*args, **kwargs)
                elif training_objective == "colbertv1":
                    return ColPali.from_pretrained(*args, **kwargs)
                else:
                    raise ValueError(f"Training objective {training_objective} not recognized")
            else:
                raise ValueError(f"Model `{pretrained_model_name_or_path}` not supported")

else:
    raise ModuleNotFoundError("Transformers must be loaded")
