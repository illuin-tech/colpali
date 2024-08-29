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
> If you want to try the pre-trained ColPali on your own documents, you should use the [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) repository. It comes with a Python package and a CLI tool for convenient evaluation.

## Associated Paper

**ColPali: Efficient Document Retrieval with Vision Language Models**
Manuel Faysse\*, Hugues Sibille\*, Tony Wu\* Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (\*Equal Contribution)

This repository contains the code used for training the vision retrievers in the paper. In particular, it contains the code for training the ColPali model, which is a vision retriever based on the ColBERT architecture.

## Setup

We used Python 3.11.6 and PyTorch 2.2.2 to train and test our models, but the codebase is expected to be compatible with Python >=3.9 and recent PyTorch versions.

The eval codebase depends on a few Python packages, which can be downloaded using the following command:

```bash
pip install colpali-engine
```

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies to use the training script for ColPali. You can do this using the following command:

```bash
pip install "colpali-engine[train]"
```

## Usage

The `scripts/` directory contains scripts to run training and inference.

### Inference

While there is an inference script in this repository, it's recommended to run inference using the  [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) package.

### Training

All the model configs used can be found in `scripts/configs/` and rely on the [configue](https://github.com/illuin-tech/configue) package for straightforward configuration. They should be used with the `train_colbert.py` script.

**Example 1: Local training**

```bash
USE_LOCAL_DATASET=0 python scripts/train/train_colbert.py scripts/configs/siglip/train_siglip_model_debug.yaml
```

or using `accelerate`:

```bash
accelerate launch scripts/train/train_colbert.py scripts/configs/train_colidefics_model.yaml
```

**Example 2: Training on a SLURM cluster**

```bash
sbatch --nodes=1 --cpus-per-task=16 --mem-per-cpu=32GB --time=20:00:00 --gres=gpu:1  -p gpua100 --job-name=colidefics --output=colidefics.out --error=colidefics.err --wrap="accelerate launch scripts/train/train_colbert.py  scripts/configs/train_colidefics_model.yaml"

sbatch --nodes=1  --time=5:00:00 -A cad15443 --gres=gpu:8  --constraint=MI250 --job-name=colpali --wrap="python scripts/train/train_colbert.py scripts/configs/train_colpali_model.yaml"
```

## Paper result reproduction

To reproduce the results from the paper, you should checkout to the `v0.1.1` tag or install the corresponding `colpali-engine` package release using:

```bash
pip install colpali-engine==0.1.1
```

## Citation

**ColPali: Efficient Document Retrieval with Vision Language Models**  

- First authors: Manuel Faysse\*, Hugues Sibille\*, Tony Wu\* (\*Equal Contribution)
- Contributors: Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo

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
