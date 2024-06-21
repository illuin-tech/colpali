# ColPali: Efficient Document Retrieval with Vision Language Models


[[Blog]]()
[[Paper]]()
[[ColPali Model card]]()
[[ViDoRe Dataset card]]()
[[Colab example]]()
[[HuggingFace Space]]()


## Associated Paper

**ColPali: Efficient Document Retrieval with Vision Language Models**
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, CELINE HUDELOT, Pierre Colombo

This repository contains the code for training custom Colbert retriever models.
Notably, we train colbert with LLMs (decoders) as well as Image Language models !

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Example usage of the model is shown in the `scripts` directory.

```bash
# hackable example script to adapt
python scripts/infer/run_inference_with_python.py
```

## Training

```bash
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/siglip/train_siglip_model_debug.yaml
```

or 

```bash
accelerate launch scripts/train/train_colbert.py scripts/configs/train_colidefics_model.yaml
```

### Configurations
All training arguments can be set through a configuration file.
The configuration file is a yaml file that contains all the arguments for training.

The construction is as follows:

```python
@dataclass
class ColModelTrainingConfig:
    model: PreTrainedModel
    tr_args: TrainingArguments = None
    output_dir: str = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    add_suffix: bool = False
    processor: Idefics2Processor = None
    tokenizer: PreTrainedTokenizer = None
    loss_func: Optional[Callable] = ColbertLoss()
    dataset_loading_func: Optional[Callable] = None
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None
```
### Example

An example configuration file is:

```yaml
config:
  (): custom_colbert.utils.train_custom_colbert_models.ColModelTrainingConfig
  output_dir: !path ../../../models/without_tabfquad/train_colpali-no-register-3b-mix-448
  processor:
    () : custom_colbert.utils.wrapper.AutoProcessorWrapper
    pretrained_model_name_or_path: "./models/paligemma-3b-mix-448"
    max_length: 50
  model:
    (): custom_colbert.utils.wrapper.AutoColModelWrapper
    pretrained_model_name_or_path: "./models/paligemma-3b-mix-448"
    training_objective: "colbertv1"
    # attn_implementation: "eager"
    torch_dtype:  !ext torch.bfloat16
#    device_map: "auto"
#    quantization_config:
#      (): transformers.BitsAndBytesConfig
#      load_in_4bit: true
#      bnb_4bit_quant_type: "nf4"
#      bnb_4bit_compute_dtype:  "bfloat16"
#      bnb_4bit_use_double_quant: true

  dataset_loading_func: !ext custom_colbert.utils.dataset_transformation.load_train_set
  eval_dataset_loader: !import ../data/test_data.yaml

  max_length: 50
  run_eval: true
  add_suffix: true
  loss_func:
    (): custom_colbert.loss.colbert_loss.ColbertPairwiseCELoss
  tr_args: !import ../tr_args/default_tr_args.yaml
  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    init_lora_weights: "gaussian"
    bias: "none"
    task_type: "FEATURE_EXTRACTION"
    target_modules: '(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)'
    # target_modules: '(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)'
```


#### Local training

```bash
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/siglip/train_siglip_model_debug.yaml
```


#### SLURM

```bash
sbatch --nodes=1 --cpus-per-task=16 --mem-per-cpu=32GB --time=20:00:00 --gres=gpu:1  -p gpua100 --job-name=colidefics --output=colidefics.out --error=colidefics.err --wrap="accelerate launch scripts/train/train_colbert.py  scripts/configs/train_colidefics_model.yaml"

sbatch --nodes=1  --time=5:00:00 -A cad15443 --gres=gpu:8  --constraint=MI250 --job-name=colpali --wrap="python scripts/train/train_colbert.py scripts/configs/train_colpali_model.yaml"
```
