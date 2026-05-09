# Adding a new model family

This guide describes the expected steps for adding a new model family to
`colpali_engine.models`. A model family is a backbone-specific package such as
`qwen3`, `gemma3`, `idefics3`, or `paligemma` that exposes one or more retriever
variants.

Most families contain:

- A `Col*` late-interaction model that returns one normalized vector per token
  and is scored with MaxSim.
- Optionally, a `Bi*` dense retrieval model that pools to one normalized vector
  per input.
- One processor per variant, responsible for image/text formatting and scoring.

## 1. Choose the package layout

Create a family directory under `colpali_engine/models`:

```text
colpali_engine/models/<family>/
    __init__.py
    col<family>/
        __init__.py
        modeling_col<family>.py
        processing_col<family>.py
    bi<family>/
        __init__.py
        modeling_bi<family>.py
        processing_bi<family>.py
```

Only add `bi<family>/` if the family supports a dense bi-encoder variant.
Follow the naming style used by nearby families. For example, Qwen variants use
`ColQwen3`, `ColQwen3Processor`, `BiQwen3`, and `BiQwen3Processor`.

## 2. Implement the Col model

The `Col*` class should usually inherit from the corresponding Transformers
backbone model, not from a generic wrapper. See
`colpali_engine/models/qwen3/colqwen3/modeling_colqwen3.py` and
`colpali_engine/models/gemma3/colgemma3/modeling_colgemma.py` for current
patterns.

The class should define:

- `main_input_name = "doc_input_ids"` for Transformers compatibility.
- A retrieval projection layer, usually `self.custom_text_proj`.
- `self.dim`, the embedding size returned by the retriever head.
- `self.padding_side`, matching the processor and backbone requirements.
- `self.mask_non_image_embeddings` when the family supports image-only masking.
- A `forward` method that returns a `torch.Tensor` shaped
  `(batch_size, sequence_length, dim)`.

The forward pass should:

1. Accept the batch produced by the processor for both images and text.
2. Adapt processor-specific image tensors before calling the backbone when the
   backbone expects a flattened visual-token layout.
3. Call the parent model with `use_cache=False`, `output_hidden_states=True`,
   and `return_dict=True`.
4. Project the last hidden states with `custom_text_proj`.
5. L2-normalize the projected embeddings on the last dimension.
6. Multiply by `attention_mask.unsqueeze(-1)` so padding tokens score as zero.
7. If `mask_non_image_embeddings=True`, zero non-image token embeddings for
   image batches.

Expose patch metadata needed by interpretability when the backbone supports it:

```python
@property
def patch_size(self) -> int:
    return self.visual.config.patch_size

@property
def spatial_merge_size(self) -> int:
    return self.visual.config.spatial_merge_size
```

Adjust the properties to match the backbone config. Some models only expose
`patch_size`.

## 3. Handle checkpoint key mappings

Adapter checkpoints often contain PEFT or backbone-specific prefixes that do not
match the retriever class. Add a `_checkpoint_conversion_mapping` to the model
when needed:

```python
_checkpoint_conversion_mapping = {
    r"^base_model\.model\.custom_text_proj": "custom_text_proj",
    r"^model\.visual": "visual",
    r"^model\.language_model": "language_model",
    r"^model\.": "",
}
```

Override `from_pretrained` to pass the mapping through `key_mapping`:

```python
@classmethod
def from_pretrained(cls, *args, **kwargs):
    key_mapping = kwargs.pop("key_mapping", None)
    if key_mapping is None:
        key_mapping = dict(getattr(super(), "_checkpoint_conversion_mapping", {}))
        key_mapping.update(getattr(cls, "_checkpoint_conversion_mapping", {}))
    return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)
```

If Transformers requires registration for the model type, register the mapping
with `register_checkpoint_conversion_mapping`, as in the Qwen and ModernVBert
implementations.

Add tests to `tests/models/test_checkpoint_key_mappings.py` for every custom
mapping that rewrites adapter keys.

## 4. Implement the processor

Processors should inherit from `BaseVisualRetrieverProcessor` and the matching
Transformers processor:

```python
class ColNewFamilyProcessor(BaseVisualRetrieverProcessor, NewFamilyProcessor):
    ...
```

The processor must implement:

- `process_images(self, images)`: converts PIL images to model-ready batches.
- `process_texts(self, texts)`: converts text inputs to model-ready batches.
- `score(self, qs, ps, device=None, **kwargs)`: delegates to
  `score_multi_vector` for `Col*` models.
- `get_n_patches(...)`: returns `(n_patches_x, n_patches_y)` for
  interpretability.

Set prompt and token attributes when the backbone needs them:

```python
visual_prompt_prefix = "..."
query_prefix = "..."
query_augmentation_token = "..."
image_token = "..."
```

Use the backbone's chat template or special tokens consistently with the
checkpoint used for training. Also set `self.tokenizer.padding_side` in
`__init__` when the model requires left or right padding.

If the processor pads per-image visual tensors for distributed training, the
model forward pass must undo that padding before calling the backbone. The Qwen
processors and models are the reference pattern for this.

## 5. Implement an optional Bi model

Add a `Bi*` model when the family needs dense single-vector retrieval. The class
normally shares the same backbone and processor conventions, but the forward pass
returns `(batch_size, hidden_size_or_dim)` instead of per-token embeddings.

Support the local pooling styles used elsewhere when possible:

- `cls`: first token.
- `last`: last token.
- `mean`: attention-mask-weighted mean.

Normalize the pooled embedding before returning it. Its processor `score` method
should delegate to `score_single_vector`.

## 6. Export the new classes

Wire the imports at every package level:

```python
# colpali_engine/models/<family>/col<family>/__init__.py
from .modeling_col<family> import ColNewFamily
from .processing_col<family> import ColNewFamilyProcessor

# colpali_engine/models/<family>/__init__.py
from .col<family> import ColNewFamily, ColNewFamilyProcessor
from .bi<family> import BiNewFamily, BiNewFamilyProcessor

# colpali_engine/models/__init__.py
from .<family> import BiNewFamily, BiNewFamilyProcessor, ColNewFamily, ColNewFamilyProcessor
```

Keep these exports stable because users import models directly from
`colpali_engine.models`.

## 7. Add tests

Create tests under `tests/models/<family>/<variant>/`, following existing model
families.

Processor tests should verify:

- `from_pretrained` returns the custom processor class.
- `process_images` returns expected keys and tensor batch dimensions.
- `process_texts` returns text tensors with the expected batch size.
- `process_queries` remains compatible with the legacy evaluator path.

Model tests should verify:

- `from_pretrained` returns the custom model class.
- Image forward pass returns a tensor with shape
  `(batch_size, sequence_length, model.dim)` for `Col*`.
- Query forward pass returns the same embedding dimension.
- Retrieval smoke tests rank matching image/query pairs correctly when a small
  public checkpoint is available.

Use `@pytest.mark.slow` for tests that download or run full checkpoints.

Run the targeted tests before opening a PR:

```bash
pytest tests/models/<family>
pytest tests/models/test_checkpoint_key_mappings.py
```

Run the linter before submitting:

```bash
ruff check .
```

## 8. Add training and example entry points when needed

If the family is trainable from this repository, add a config under
`scripts/configs/<family>/` and update any training scripts that need to import
the new classes. Keep the config names aligned with the model class names, for
example `train_col<family>_model.py` or `train_col<family>_model.yaml`.

If interpretability is supported, add an example under
`examples/interpretability/<variant>/` and make sure `get_n_patches` plus
`get_image_mask` return masks in the same token order as the model embeddings.

## 9. Update user-facing documentation

When the checkpoint is public and supported, update the model table in
`README.md` with:

- The Hugging Face model id.
- The base backbone.
- The license.
- Notes about dynamic resolution, embedding dimension, or masking behavior.
- Whether the model is currently supported.

Add usage snippets only if loading or preprocessing differs from the existing
quick start pattern.

## Review checklist

Before submitting the change, check that:

- The model and processor can be imported from `colpali_engine.models`.
- `process_images`, `process_texts`, and `process_queries` all work.
- `model(**processor.process_images(...))` and
  `model(**processor.process_queries(...))` return normalized tensors.
- Padding embeddings are zeroed for `Col*` outputs.
- Checkpoint mappings load LoRA or adapter checkpoints without manual key edits.
- Slow tests are marked, and fast tests do not download large checkpoints unless
  the existing family tests already do the same.
