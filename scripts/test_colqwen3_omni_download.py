from colpali_engine.models.qwen_omni import ColQwen3Omni, ColQwen3OmniProcessor


MODEL_NAME = "BidirLM/BidirLM-Omni-2.5B-Embedding"


def main() -> None:
    processor = ColQwen3OmniProcessor.from_pretrained(MODEL_NAME)
    model = ColQwen3Omni.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype="auto",
    )

    print(f"Processor: {type(processor).__name__}")
    print(f"Model: {type(model).__name__}")
    print(f"Model type: {model.config.model_type}")
    print(f"Device: {model.device}")


if __name__ == "__main__":
    main()
