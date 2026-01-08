"""
CLI to load a `colpali_engine` model from Hugging Face and save it.

Example:
    python scripts/init_base_model.py \
        --model-class colpali_engine.models.qwen3.colqwen3.ColQwen3 \
        --model-name-or-path Qwen/Qwen3-VL-2B-Instruct \
        --save-path ./models/colqwen3 \
        --hf-repo vidore/colqwen3-base
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Type

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_class(class_path: str) -> Type[Any]:
    """Dynamically import and return a class from a fully qualified path."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to import class from {class_path}: {e}")
        sys.exit(1)


def _resolve_load_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Map CLI arguments to Transformers-style from_pretrained kwargs."""
    kwargs = {
        "pretrained_model_name_or_path": args.model_name_or_path,
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code or None,  # Only include if True
        "torch_dtype": args.dtype,
        "device_map": args.device_map,
    }
    # Filter out None values to use model defaults
    return {k: v for k, v in kwargs.items() if v is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a ColModel via from_pretrained and save it locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--model-name-or-path", required=True, help="HF model ID or local path.")
    parser.add_argument(
        "--model-class", required=True, help="Full class path (e.g., 'colpali_engine.models.ColQwen3')."
    )
    parser.add_argument("--save-path", required=True, help="Target directory for the model.")

    # Optional configuration
    parser.add_argument("--hf-repo", help="Optional HF repo to push to.")
    parser.add_argument("--revision", help="Specific model revision/branch.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from Hub.")
    parser.add_argument("--dtype", help="torch dtype (e.g., 'float16', 'bfloat16').")
    parser.add_argument("--device-map", help="Device placement (e.g., 'auto', 'cpu').")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Initialize Class
    model_class = _resolve_class(args.model_class)
    logger.info(f"Using model class: {model_class.__name__}")

    # 2. Load Model
    load_kwargs = _resolve_load_kwargs(args)
    logger.info(f"Loading model from {args.model_name_or_path}...")

    try:
        model = model_class.from_pretrained(**load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # 3. Save Locally
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)

    # 4. Optional Push
    if args.hf_repo:
        logger.info(f"Pushing to Hugging Face: {args.hf_repo}")
        model.push_to_hub(args.hf_repo, commit_message="Initial model upload")

    logger.info("Task completed successfully!")


if __name__ == "__main__":
    main()
