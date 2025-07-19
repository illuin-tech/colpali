import argparse
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True, help="where to write model + script copy")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"], help="trainer to use")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"], help="loss function to use")
    p.add_argument("--peft", action="store_true", help="use PEFT for training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.loss == "ce":
        loss_func = ColbertLoss(
            temperature=args.tau,
            normalize_scores=True,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(
            normalize_scores=False,
        )
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    import torch
    from torch import nn
    from transformers import PretrainedConfig

    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    from colpali_engine.models.asymmetric.asymmetric_model import AsymmetricModel

    query_model = ColIdefics3.from_pretrained(
        "./models/base_models/ColSmolVLM-Instruct-500M-base",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        use_cache=False,
    ).train()

    doc_model = ColIdefics3.from_pretrained(
        "./models/ColSmolVLM-Instruct-256M-base",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        use_cache=False,
    ).train()

    # Example usage
    config = PretrainedConfig()
    print(f"Query model parameters before: {sum(p.numel() for p in query_model.model.parameters())}")
    # Remove vision model in the query model
    query_model.model.vision_model = nn.Identity()

    # print num parameters in the query model
    print(f"Query model parameters after: {sum(p.numel() for p in query_model.model.parameters())}")
    # print num parameters in the document model
    print(f"Document model parameters: {sum(p.numel() for p in doc_model.model.parameters())}")

    model = AsymmetricModel(config=config, query_model=query_model, document_model=doc_model)

    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColIdefics3Processor.from_pretrained("./models/colSmol-256M"),
        model=model,
        train_dataset=ColPaliEngineDataset(
            load_dataset("./data_dir/colpali_train_set", split="train"), pos_target_column_name="image"
        ),
        eval_dataset=ColPaliEngineDataset(
            load_dataset("./data_dir/colpali_train_set", split="test"), pos_target_column_name="image"
        ),
        run_eval=True,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            dataloader_num_workers=4,
            save_steps=500,
            logging_steps=10,
            eval_steps=100,
            warmup_steps=100,
            learning_rate=args.lr,
            save_total_limit=1,
            dataloader_prefetch_factor=2,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
        ),
        peft_config=LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        )
        if args.peft
        else None,
    )

    # make sure output_dir exists and copy script for provenance
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)
    trainer.train()
    trainer.save()
