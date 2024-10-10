# ColPali: Efficient Document Retrieval with Vision Language Models 👀

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![GitHub](https://img.shields.io/badge/ViDoRe_Benchmark-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/vidore-benchmark)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/vidore)
[![GitHub](https://img.shields.io/badge/Cookbooks-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tonywu71/colpali-cookbooks)

[![Test](https://github.com/illuin-tech/colpali/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/illuin-tech/colpali/actions/workflows/test.yml)
[![Version](https://img.shields.io/pypi/v/colpali-engine?color=%2334D058&label=pypi%20package)](https://pypi.org/project/colpali-engine/)
[![Downloads](https://static.pepy.tech/badge/colpali-engine)](https://pepy.tech/project/colpali-engine)

---

[[Model card]](https://huggingface.co/vidore/colpali)
[[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)
[[Demo]](https://huggingface.co/spaces/manu/ColPali-demo)
[[Blog Post]](https://huggingface.co/blog/manu/colpali)

> [!TIP]
> For production usage in your RAG pipelines, we recommend using the [`byaldi`](https://github.com/AnswerDotAI/byaldi) package, which is a lightweight wrapper around the `colpali-engine` package developed by the author of the popular [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) repostiory. 🐭

## Associated Paper

This repository contains the code used for training the vision retrievers in the [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://arxiv.org/abs/2407.01449) paper. In particular, it contains the code for training the ColPali model, which is a vision retriever based on the ColBERT architecture and the PaliGemma model.

## Introduction

With our new model *ColPali*, we propose to leverage VLMs to construct efficient multi-vector embeddings in the visual space for document retrieval. By feeding the ViT output patches from PaliGemma-3B to a linear projection, we create a multi-vector representation of documents. We train the model to maximize the similarity between these document embeddings and the query embeddings, following the ColBERT method.

Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

![ColPali Architecture](assets/colpali_architecture.webp)

## List of ColVision models

| Model                                                        | Score on [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) 🏆 | License    | Comments                                                     | Currently supported |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ | ------------------- |
| [vidore/colpali](https://huggingface.co/vidore/colpali)      | 81.3                                                         | Gemma      | • Based on `google/paligemma-3b-mix-448`.<br />• Checkpoint used in the ColPali paper. | ❌                   |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1) | 81.5                                                         | Gemma      | • Based on `google/paligemma-3b-mix-448`.                    | ✅                   |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) | 83.1                                                         | Gemma      | • Based on `google/paligemma-3b-mix-448`.                    | ✅                   |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 86.6                                                         | Apache 2.0 | • Based on `Qwen/Qwen2-VL-2B-Instruct`.<br />• Supports dynamic resolution.<br />• Trained using 768 image patches per page. | ✅                   |

## Setup

We used Python 3.11.6 and PyTorch 2.2.2 to train and test our models, but the codebase is compatible with Python >=3.9 and recent PyTorch versions. To install the package, run:

```bash
pip install colpali-engine
```

> [!WARNING]
> For ColPali versions above v1.0, make sure to install the `colpali-engine` package from source or with a version above v0.2.0.

## Usage

### Quick start

```python
import torch
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor

model_name = "vidore/colpali-v1.2"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "Is attention really all you need?",
    "Are Benjamin, Antoine, Merve, and Jo best friends?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)

```

### Inference

You can find an example [here](https://github.com/illuin-tech/colpali/blob/main/scripts/infer/run_inference_with_python.py). If you need an indexing system, we recommend using [`byaldi`](https://github.com/AnswerDotAI/byaldi) - [RAGatouille](https://github.com/AnswerDotAI/RAGatouille)'s little sister 🐭 - which share a similar API and leverages our `colpali-engine` package.

### Benchmarking

To benchmark ColPali to reproduce the results on the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard), it is recommended to use the [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) package.

### Interpretability with similarity maps

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones.

You can generate similarity maps using the `generate-similarity-maps`. For instance, you can reproduce the similarity maps from the paper using the images from [`data/interpretability_examples`](https://github.com/illuin-tech/vidore-benchmark/tree/main/data/interpretability_examples) and by running the following command. You can also feed multiple documents and queries at once to generate multiple similarity maps.

```bash
generate-similarity-maps \
    --model-name "vidore/colpali-v1.2" \
    --documents "data/interpretability_examples/shift_kazakhstan.jpg" \
    --queries "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?" \
    --documents "data/interpretability_examples/energy_electricity_generation.jpeg" \
    --queries "Which hour of the day had the highest overall electricity generation in 2019?"
```

> [!NOTE]
> The current version of `vidore-benchmark` uses a different ColPali checkpoint than the one used in the paper. As a result, the similarity maps may differ slightly from the ones presented in the paper.

### Training

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies to use the training script for ColPali. You can do this using the following command:

```bash
pip install "colpali-engine[train]"
```

All the model configs used can be found in `scripts/configs/` and rely on the [configue](https://github.com/illuin-tech/configue) package for straightforward configuration. They should be used with the `train_colbert.py` script.

#### Example 1: Local training

```bash
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml
```

or using `accelerate`:

```bash
accelerate launch scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml
```

#### Example 2: Training on a SLURM cluster

```bash
sbatch --nodes=1 --cpus-per-task=16 --mem-per-cpu=32GB --time=20:00:00 --gres=gpu:1  -p gpua100 --job-name=colidefics --output=colidefics.out --error=colidefics.err --wrap="accelerate launch scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml"

sbatch --nodes=1  --time=5:00:00 -A cad15443 --gres=gpu:8  --constraint=MI250 --job-name=colpali --wrap="python scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml"
```

## Paper result reproduction

To reproduce the results from the paper, you should checkout to the `v0.1.1` tag or install the corresponding `colpali-engine` package release using:

```bash
pip install colpali-engine==0.1.1
```

## Citation

**ColPali: Efficient Document Retrieval with Vision Language Models**  

Authors: **Manuel Faysse**\*, **Hugues Sibille**\*, **Tony Wu**\*, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (\* denotes equal contribution)

```latex
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}
```
