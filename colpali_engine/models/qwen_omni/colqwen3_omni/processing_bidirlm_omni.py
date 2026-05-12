from typing import Optional, Union

import numpy as np
from transformers.audio_utils import AudioInput

try:
    from torchcodec.decoders import AudioDecoder
except ImportError:
    AudioDecoder = None

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)


def _resample_audio(arr: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "`librosa` is required to resample audio inputs. Install it or pass audio with the processor's native "
            f"sampling rate ({target_sr})."
        ) from exc
    return librosa.resample(arr, orig_sr=orig_sr, target_sr=target_sr)


# ── Kwargs classes ─────────────────────────────────────────────────────────


class BidirLMOmniVideosKwargs(VideosKwargs, total=False):
    pass


class BidirLMOmniImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class BidirLMOmniProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: BidirLMOmniImagesKwargs
    videos_kwargs: BidirLMOmniVideosKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "right",
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


# ── Audio helpers ──────────────────────────────────────────────────────────


def _get_feat_extract_output_lengths(input_lengths):
    """Computes the output length of the audio encoder's convolutional layers.
    Three Conv2d layers each with kernel=3, stride=2, padding=1.
    Per-layer formula: floor((L - 1) / 2) + 1
    """
    length = (input_lengths - 1) // 2 + 1
    length = (length - 1) // 2 + 1
    length = (length - 1) // 2 + 1
    return length


# ── Processor ──────────────────────────────────────────────────────────────


class BidirLMOmniProcessor(ProcessorMixin):
    attributes = ["image_processor", "video_processor", "feature_extractor", "tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        max_image_size: Optional[int] = None,
    ):
        super().__init__(
            image_processor,
            video_processor,
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
        )

        if max_image_size is not None and image_processor is not None:
            max_pixels = max_image_size * max_image_size
            image_processor.size["longest_edge"] = max_pixels
            if image_processor.size["shortest_edge"] > max_pixels:
                image_processor.size["shortest_edge"] = max_pixels

        # ── Vision tokens (from Qwen3VLProcessor) ─────────────────────
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None) is not None
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None) is not None
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )

        # ── Audio tokens (from Qwen3ASRProcessor) ─────────────────────
        self.audio_token = "<|audio_pad|>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
        self.audio_bos_token = (
            "<|audio_start|>" if not hasattr(tokenizer, "audio_bos_token") else tokenizer.audio_bos_token
        )
        self.audio_eos_token = (
            "<|audio_end|>" if not hasattr(tokenizer, "audio_eos_token") else tokenizer.audio_eos_token
        )

        self.sampling_rate = self.feature_extractor.sampling_rate

    # ── __call__ ───────────────────────────────────────────────────────

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[BidirLMOmniProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for the model. Processes text with the tokenizer, images with
        the image processor, videos with the video processor, and audio with the
        WhisperFeatureExtractor.

        Args:
            images: PIL images, numpy arrays, or tensors.
            text: Text sequences to encode.
            videos: Video arrays (4D) or nested lists of frames.
            audio: Audio numpy arrays.
        """
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            BidirLMOmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # ── Image processing (from Qwen3VLProcessor) ──────────────────
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # ── Video processing (from Qwen3VLProcessor) ──────────────────
        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        # ── Audio processing (from Qwen3ASRProcessor) ─────────────────
        if audio is not None:
            pipeline_sr = output_kwargs["audio_kwargs"].get("sampling_rate", self.sampling_rate)
            if not isinstance(audio, (list, tuple)):
                audio = [audio]
            audio = [self._normalize_audio(a, pipeline_sr) for a in audio]

            output_kwargs["audio_kwargs"]["sampling_rate"] = self.sampling_rate
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["truncation"] = False
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        # ── Token expansion ────────────────────────────────────────────
        if not isinstance(text, list):
            text = [text]

        text = text.copy()

        # Image placeholder expansion
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Video placeholder expansion
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "BiQwen3VL requires frame timestamps to construct prompts, but the `fps` of the input "
                            "video could not be inferred. Defaulting to `fps=24`."
                        )
                        metadata.fps = 24
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )
                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}",
                            video_placeholder,
                            1,
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # Audio placeholder expansion
        text = self._replace_audio_special_tokens(text, audio_lengths)

        # ── Tokenize ──────────────────────────────────────────────────
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **audio_inputs},
            tensor_type=return_tensors,
        )

    # ── Audio token expansion ──────────────────────────────────────────

    def _replace_audio_special_tokens(self, text, audio_lengths):
        """Replace audio placeholder tokens with the correct number of pad tokens."""
        processed_text = []
        for sample in text:
            while self.audio_token in sample:
                sample = sample.replace(
                    self.audio_token,
                    "<|audio_placeholder|>" * next(audio_lengths),
                    1,
                )
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            processed_text.append(sample)
        return processed_text

    # ── Video timestamp calculation (from Qwen3VLProcessor) ────────────

    def _calculate_timestamps(self, indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    # ── Audio chunking helper (from Qwen3ASRProcessor) ─────────────────

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """Splits token index list into chunks based on token value ranges."""

        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    # ── Post processing ────────────────────────────────────────────────

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _normalize_audio(self, a, pipeline_sr=None):
        """Normalize a single audio item to a float32 numpy array at self.sampling_rate.

        Accepts:
        - list[float]  — plain Python list of samples
        - np.ndarray   — raw samples (resampled via pipeline_sr if it differs from self.sampling_rate)
        - dict         — HuggingFace datasets Audio dict {"array": ..., "sampling_rate": ...}
        - AudioDecoder — datasets 4.x lazy decoder (torchcodec); None if torchcodec is not installed
        """
        if isinstance(a, (list, np.ndarray)):
            arr = np.asarray(a, dtype=np.float32)
            if pipeline_sr and pipeline_sr != self.sampling_rate:
                arr = _resample_audio(arr, orig_sr=pipeline_sr, target_sr=self.sampling_rate)
        elif isinstance(a, dict):  # HuggingFace datasets Audio dict: {"array": ..., "sampling_rate": ...}
            arr = np.asarray(a["array"], dtype=np.float32)
            src_sr = a.get("sampling_rate")
            if src_sr and src_sr != self.sampling_rate:
                arr = _resample_audio(arr, orig_sr=src_sr, target_sr=self.sampling_rate)
        elif AudioDecoder is not None and isinstance(a, AudioDecoder):
            samples = a.get_all_samples()
            arr = samples.data.float().mean(dim=0).cpu().numpy()
            src_sr = samples.sample_rate
            if src_sr and src_sr != self.sampling_rate:
                arr = _resample_audio(arr, orig_sr=src_sr, target_sr=self.sampling_rate)
        else:
            raise TypeError(
                f"Unsupported audio type: {type(a).__name__}. "
                "Expected a plain list, a numpy array, a HuggingFace datasets Audio dict, "
                "or a torchcodec AudioDecoder."
            )
        return arr

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        # Normalize audio in user turn content items to numpy arrays before the base
        # class processes them. Accepts lists, numpy arrays, HF Audio dicts, or
        # AudioDecoder objects. Only user turns can contain audio.
        # Accept both single conversation (List[Dict]) and batch (List[List[Dict]]).
        # Build a local batch view for safe traversal WITHOUT changing what is passed
        # to super(), so super's return type (str for single, List[str] for batch)
        # is preserved exactly as the caller expects.
        batch = [conversations] if (conversations and isinstance(conversations[0], dict)) else conversations
        for conv in batch:
            for turn in conv:
                if turn.get("role") != "user":
                    continue
                for item in turn.get("content", []):
                    audio_val = item.get("audio")
                    if item.get("type") == "audio" and not isinstance(audio_val, np.ndarray):
                        item["audio"] = self._normalize_audio(audio_val)
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))


__all__ = ["BidirLMOmniProcessor"]
