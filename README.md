# ColPali: Efficient Document Retrieval with Vision Language Models 👀

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![GitHub](https://img.shields.io/badge/ViDoRe_Benchmark-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/vidore-benchmark)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/vidore)

[[Model card]](https://huggingface.co/vidore/colpali)
[[ViDoRe Benchmark]](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d)
[[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)
[[Demo]](https://huggingface.co/spaces/manu/ColPali-demo)
[[Blog Post]](https://huggingface.co/blog/manu/colpali)

> [!TIP]
> For production usage in your RAG pipelines, we recommend using the [`byaldi`](https://github.com/AnswerDotAI/byaldi) package, which is a lightweight wrapper around the `colpali-engine` package developed by the author of the popular [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) repostiory. 🐭

## Associated Paper

This repository contains the code used for training the vision retrievers in the [**ColPali: Efficient Document Retrieval with Vision Language Models**](https://arxiv.org/abs/2407.01449) paper. In particular, it contains the code for training the ColPali model, which is a vision retriever based on the ColBERT architecture.

## Setup

We used Python 3.11.6 and PyTorch 2.2.2 to train and test our models, but the codebase is compatible with Python >=3.9 and recent PyTorch versions.

The eval codebase depends on a few Python packages, which can be downloaded using the following command:

```bash
pip install colpali-engine
```

> [!WARNING]
> For ColPali versions above v1.0, make sure to install the `colpali-engine` package from source or with a version above v0.2.0.

## Usage

### Inference

This repository doesn't contain the code to run optimized retrieval for RAG pipelines. For this, we recommend using [`byaldi`](https://github.com/AnswerDotAI/byaldi) - [RAGatouille](https://github.com/AnswerDotAI/RAGatouille)'s little sister 🐭 - which share a similar API and leverages our `colpali-engine` package.

### Benchmarking

To benchmark ColPali to reproduce the results on the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard), it is recommended to use the [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) package.

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

Authors: Manuel Faysse\*, Hugues Sibille\*, Tony Wu\*, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo

(\* Denotes Equal Contribution)

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
