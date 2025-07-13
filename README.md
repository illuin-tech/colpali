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

## Associated Paper

This repository contains the code used for training the vision retrievers in the [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://arxiv.org/abs/2407.01449) paper. In particular, it contains the code for training the ColPali model, which is a vision retriever based on the ColBERT architecture and the PaliGemma model.

## Introduction

With our new model *ColPali*, we propose to leverage VLMs to construct efficient multi-vector embeddings in the visual space for document retrieval. By feeding the ViT output patches from PaliGemma-3B to a linear projection, we create a multi-vector representation of documents. We train the model to maximize the similarity between these document embeddings and the query embeddings, following the ColBERT method.

Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

![ColPali Architecture](assets/colpali_architecture.webp)

## List of ColVision models

| Model                                                               | Score on [ViDoRe](https://huggingface.co/spaces/vidore/vidore-leaderboard) 🏆 | License    | Comments                                                                                                                                                       | Currently supported |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| [vidore/colpali](https://huggingface.co/vidore/colpali)             | 81.3                                                                          | Gemma      | • Based on `google/paligemma-3b-mix-448`.<br />• Checkpoint used in the ColPali paper.                                                                         | ❌                   |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1)   | 81.5                                                                          | Gemma      | • Based on `google/paligemma-3b-mix-448`.<br />• Fix right padding for queries.                                                                                | ✅                   |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)   | 83.9                                                                          | Gemma      | • Similar to `vidore/colpali-v1.1`.                                                                                                                            | ✅                   |
| [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3)   | 84.8                                                                          | Gemma      | • Similar to `vidore/colpali-v1.2`.<br />• Trained with a larger effective batch size of 256 batch size for 3 epochs.                                          | ✅                   |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 87.3                                                                          | Apache 2.0 | • Based on `Qwen/Qwen2-VL-2B-Instruct`.<br />• Supports dynamic resolution.<br />• Trained using 768 image patches per page and an effective batch size of 32. | ✅                   |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | 89.3                                                                          | Apache 2.0 | • Similar to `vidore/colqwen2-v0.1`, but trained with more powerful GPUs and with a larger effective batch size (256).                                         | ✅                   |
| [vidore/colqwen2.5-v0.1](https://huggingface.co/vidore/colqwen2.5-v0.1) | 88.8                                                                          | Apache 2.0 | • Based on `Qwen/Qwen2 5-VL-3B-Instruct`<br />• Supports dynamic resolution.<br />• Trained using 768 image patches per page and an effective batch size of 32.                                         | ✅                   |
| [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | 89.4                                                                          | Apache 2.0 | • Similar to `vidore/colqwen2.5-v0.1`, but trained with slightly different hyper parameters                                        | ✅                   |
| [vidore/colSmol-256M](https://huggingface.co/vidore/colSmol-256M)   | 80.1                                                                          | Apache 2.0 | • Based on `HuggingFaceTB/SmolVLM-256M-Instruct`.                                                                                                              | ✅                   |
| [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M)   | 82.3                                                                          | Apache 2.0 | • Based on `HuggingFaceTB/SmolVLM-500M-Instruct`.                                                                                                              | ✅                   |

## Setup

We used Python 3.11.6 and PyTorch 2.4 to train and test our models, but the codebase is compatible with Python >=3.9 and recent PyTorch versions. To install the package, run:

```bash
pip install colpali-engine # from PyPi
pip install git+https://github.com/illuin-tech/colpali # from source
```

Mac users using MPS with the ColQwen models have reported errors with torch 2.6.0. These errors are fixed by downgrading to torch 2.5.1.

> [!WARNING]
> For ColPali versions above v1.0, make sure to install the `colpali-engine` package from source or with a version above v0.2.0.

## Usage

### Quick start

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
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

We now support `fast-plaid` experimentally to make matching quicker for larger corpus sizes:

```python
# !pip install --no-deps fast-plaid fastkmeans

# Process the inputs by batches of 4
dataloader = DataLoader(
    dataset=images,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: processor.process_images(x),
)

ds  = []
for batch_doc in tqdm(dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

plaid_index = processor.create_plaid_index(ds)

scores = processor.get_topk_plaid(query_embeddings, plaid_index, k=10)
```

### Benchmarking

To benchmark ColPali on the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard), use the [`vidore-benchmark`](https://github.com/illuin-tech/vidore-benchmark) package.

### Interpretability with similarity maps

By superimposing the late interaction similarity maps on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones.

To use the `interpretability` module, you need to install the `colpali-engine[interpretability]` package:

```bash
pip install colpali-engine[interpretability]
```

Then, after generating your embeddings with ColPali, use the following code to plot the similarity maps for each query token:

<details>
<summary><strong>🔽 Click to expand code snippet</strong></summary>

```python
import torch
from PIL import Image

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

model_name = "vidore/colpali-v1.3"
device = get_torch_device("auto")

# Load the model
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

# Load the processor
processor = ColPaliProcessor.from_pretrained(model_name)

# Load the image and query
image = Image.open("shift_kazakhstan.jpg")
query = "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"

# Preprocess inputs
batch_images = processor.process_images([image]).to(device)
batch_queries = processor.process_queries([query]).to(device)

# Forward passes
with torch.no_grad():
    image_embeddings = model.forward(**batch_images)
    query_embeddings = model.forward(**batch_queries)

# Get the number of image patches
n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size)

# Get the tensor mask to filter out the embeddings that are not related to the image
image_mask = processor.get_image_mask(batch_images)

# Generate the similarity maps
batched_similarity_maps = get_similarity_maps_from_embeddings(
    image_embeddings=image_embeddings,
    query_embeddings=query_embeddings,
    n_patches=n_patches,
    image_mask=image_mask,
)

# Get the similarity map for our (only) input image
similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

# Tokenize the query
query_tokens = processor.tokenizer.tokenize(query)

# Plot and save the similarity maps for each query token
plots = plot_all_similarity_maps(
    image=image,
    query_tokens=query_tokens,
    similarity_maps=similarity_maps,
)
for idx, (fig, ax) in enumerate(plots):
    fig.savefig(f"similarity_map_{idx}.png")
```

</details>

For a more detailed example, you can refer to the interpretability notebooks from the [ColPali Cookbooks 👨🏻‍🍳](https://github.com/tonywu71/colpali-cookbooks) repository.

### Token pooling

[Token pooling](https://doi.org/10.48550/arXiv.2409.14683) is a CRUDE-compliant method (document addition/deletion-friendly) that aims at reducing the sequence length of multi-vector embeddings. For ColPali, many image patches share redundant information, e.g. white background patches. By pooling these patches together, we can reduce the amount of embeddings while retaining most of the page's signal. Retrieval performance with hierarchical mean token pooling on image embeddings can be found in the [ColPali paper](https://doi.org/10.48550/arXiv.2407.01449). In our experiments, we found that a pool factor of 3 offered the optimal trade-off: the total number of vectors is reduced by $66.7\%$ while $97.8\%$ of the original performance is maintained.

To use token pooling, you can use the `HierarchicalEmbeddingPooler` class from the `colpali-engine` package:

<details>
<summary><strong>🔽 Click to expand code snippet</strong></summary>

```python
import torch

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

# Dummy multivector embeddings
list_embeddings = [
    torch.rand(10, 768),
    torch.rand(20, 768),
]

# Define the pooler with the desired level of compression
pooler = HierarchicalTokenPooler()

# Pool the embeddings
outputs = pooler.pool_embeddings(list_embeddings, pool_factor=2)
```

If your inputs are padded 3D tensor embeddings instead of lists of 2D tensors, use `padding=True` and specify the padding used by your tokenizer to make sure the `HierarchicalTokenPooler` correctly removes the padding values before pooling:

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
from colpali_engine.models import ColQwen2, ColQwen2Processor

model_name = "vidore/colqwen2-v1.0"
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(model_name)

token_pooler = HierarchicalTokenPooler()

# Your page images
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (32, 32), color="black"),
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)

# Apply token pooling (reduces the sequence length of the multi-vector embeddings)
image_embeddings = token_pooler.pool_embeddings(
    image_embeddings,
    pool_factor=2,
    padding=True,
    padding_side=processor.tokenizer.padding_side,
)
```

</details>

### Training

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies to use the training script for ColPali. You can do this using the following command:

```bash
pip install "colpali-engine[train]"
```

All the model configs used can be found in `scripts/configs/` and rely on the [configue](https://github.com/illuin-tech/configue) package for straightforward configuration. They should be used with the `train_colbert.py` script.

<details>
<summary><strong>🔽 Example 1: Local training</strong></summary>


```bash
accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py
```

</details>

<details>
<summary><strong>🔽 Example 2: Training on a SLURM cluster</strong></summary>

```bash
sbatch --nodes=1 --cpus-per-task=16 --mem-per-cpu=32GB --time=20:00:00 --gres=gpu:1  -p gpua100 --job-name=colidefics --output=colidefics.out --error=colidefics.err --wrap="accelerate launch scripts/train/train_colbert.py scripts/configs/pali/train_colpali_docmatix_hardneg_model.yaml"

sbatch --nodes=1  --time=5:00:00 -A cad15443 --gres=gpu:8  --constraint=MI250 --job-name=colpali --wrap="accelerate launch --multi-gpu scripts/configs/qwen2/train_colqwen25_model.py"
```

</details>

## Contributing

We welcome contributions to ColPali! 🤗

To contribute to ColPali, first install the development dependencies for proper testing/linting:

```bash
pip install "colpali-engine[dev]"
```

To run all the tests, you will have to install all optional dependencies (or you'll get an error in test discovery):

```bash
pip install "colpali-engine[all]"
```

When your PR is ready, ping one of the repository maintainers. We will do our best to review it as soon as possible!

## Community Projects

Several community projects and ressources have been developed around ColPali to facilitate its usage. Feel free to reach out if you want to add your project to this list!

<details>
<summary><strong>🔽 Libraries 📚</strong></summary>

| Library Name  | Description                                                                                                                                                                                                                                          |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
| Byaldi        | [`Byaldi`](https://github.com/AnswerDotAI/byaldi) is [RAGatouille](https://github.com/AnswerDotAI/RAGatouille)'s equivalent for ColPali, leveraging the `colpali-engine` package to facilitate indexing and storing embeddings.                      |
| PyVespa       | [`PyVespa`](https://pyvespa.readthedocs.io/en/latest/examples/colpali-document-retrieval-vision-language-models-cloud.html) allows interaction with [Vespa](https://vespa.ai/), a production-grade vector database, with detailed ColPali support.   |
| Qdrant | Tutorial about using ColQwen2 with the [Qdrant](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/) vector database. |
| Elastic Search     | Tutorial about using ColPali with the [Elastic Search](https://www.elastic.co/search-labs/blog/elastiacsearch-colpali-document-search) vector database. |
| Weaviate | Tutorial about using multi-vector embeddings with the [Weaviate](https://weaviate.io/developers/weaviate/tutorials/multi-vector-embeddings) vector database. |
| Candle        | [Candle](https://github.com/huggingface/candle/tree/main/candle-examples/examples/colpali) enables ColPali inference with an efficient ML framework for Rust.                                                                                        |
| EmbedAnything | [`EmbedAnything`](https://github.com/StarlightSearch/EmbedAnything) Allows end-to-end ColPali inference with both Candle and ONNX backend.                                                                                                           |
| DocAI         | [DocAI](https://github.com/PragmaticMachineLearning/docai) uses ColPali with GPT-4o and Langchain to extract structured information from documents.                                                                                                  |
| VARAG         | [VARAG](https://github.com/adithya-s-k/VARAG) uses ColPali in a vision-only and a hybrid RAG pipeline.                                                                                                                                               |
| ColBERT Live! | [`ColBERT Live!`](https://github.com/jbellis/colbert-live/) enables ColPali usage with vector databases supporting large datasets, compression, and non-vector predicates.                                                                           |
| ColiVara      | [`ColiVara`](https://github.com/tjmlabs/ColiVara/) is retrieval API that allows you to store, search, and retrieve documents based on their visual embedding. It is a web-first implementation of the ColPali paper using ColQwen2 as the LLM model. |
| BentoML       | Deploy ColPali easily with BentoML using [this example repository](https://github.com/bentoml/BentoColPali). BentoML features adaptive batching and zero-copy I/O to minimize overhead.                                                              |
| NoOCR       | NoOCR is end-to-end, [open source](https://github.com/kyryl-opens-ml/no-ocr) solution for complex PDFs, powered by ColPali embeddings. |
| Astra Multi-vector     | [`Astra-multivector`](https://github.com/brian-ogrady/astradb-multivector) provides enterprise-grade integration with AstraDB for late-interaction models like ColPali, ColQwen2, and ColBERT. It implements efficient token pooling and embedding caching strategies to dramatically reduce latency and index size while maintaining retrieval quality. The library leverages Cassandra's distributed architecture for high-throughput vector search at scale. |
| Mixpeek       | [Mixpeek](https://docs.mixpeek.com/processing/feature-extractors) is a production platform for multimodal late-interaction retrieval. It supports models like ColBERT, ColPaLI, and ColQwen2 with built-in indexing, versioning, A/B testing, and explainability across image, text, video, and PDF pipelines. |


</details>

<details>
<summary><strong>🔽 Notebooks 📙</strong></summary>

| Notebook Title                                               | Author & Link                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ColPali Cookbooks                                            | [Tony's Cookbooks (ILLUIN)](https://github.com/tonywu71/colpali-cookbooks) 🙋🏻 |
| Vision RAG Tutorial                                          | [Manu's Vision Rag Tutorial (ILLUIN)](https://github.com/ManuelFay/Tutorials/blob/main/Tuesday_Practical_2_Vision_RAG.ipynb) 🙋🏻 |
| ColPali (Byaldi) + Qwen2-VL for RAG                          | [Merve's Notebook (HuggingFace 🤗)](https://github.com/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb) |
| Indexing ColPali with Qdrant                                 | [Daniel's Notebook (HuggingFace 🤗)](https://danielvanstrien.xyz/posts/post-with-code/colpali-qdrant/2024-10-02_using_colpali_with_qdrant.html) |
| Weaviate Tutorial                                            | [Connor's ColPali POC (Weaviate)](https://github.com/weaviate/recipes/blob/main/weaviate-features/named-vectors/NamedVectors-ColPali-POC.ipynb) |
| Use ColPali for Multi-Modal Retrieval with Milvus            | [Milvus Documentation](https://milvus.io/docs/use_ColPali_with_milvus.md) |
| Data Generation                                              | [Daniel's Notebook (HuggingFace 🤗)](https://danielvanstrien.xyz/posts/post-with-code/colpali/2024-09-23-generate_colpali_dataset.html) |
| Finance Report Analysis with ColPali and Gemini              | [Jaykumaran (LearnOpenCV)](https://github.com/spmallick/learnopencv/tree/master/Multimodal-RAG-with-ColPali-Gemini) |
| Multimodal Retrieval-Augmented Generation (RAG) with Document Retrieval (ColPali) and Vision Language Models (VLMs) | [Sergio Paniego](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms) |
| Document Similarity Search with ColPali                      | [Frank Sommers](https://colab.research.google.com/github/fsommers/documentai/blob/main/Document_Similarity_with_ColPali_0_2_2_version.ipynb) |
| End-to-end ColPali inference with EmbedAnything              | [Akshay Ballal (EmbedAnything)](https://colab.research.google.com/drive/1-Eiaw8wMm8I1n69N1uKOHkmpw3yV22w8?usp=sharing) |
| ColiVara: A ColPali Retrieval API                            | [A simple RAG Example](https://github.com/tjmlabs/ColiVara-docs/blob/main/cookbook/RAG.ipynb) |
| Multimodal RAG with Document Retrieval (ColPali), Vision Language Model (ColQwen2) and Amazon Nova | [Suman's Notebook (AWS)](https://github.com/debnsuma/fcc-ai-engineering-aws/blob/main/05-multimodal-rag-with-colpali/01-multimodal-retrival-with-colpali-retreve-gen.ipynb) |
| Multi-vector RAG: Using Weaviate to search a collection of PDF documents | [Weaviate's Notebook](https://github.com/weaviate/recipes/blob/main/weaviate-features/multi-vector/multi-vector-colipali-rag.ipynb) |

</details>

<details>
<summary><strong>🔽 Other resources</strong></summary>

- 📝 = blog post
- 📋 = PDF / slides
- 📹 = video

| Title                                                                                    | Author & Link                                                                                                                                                 |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| State of AI report 2024                                                                  | [Nathan's report](https://www.stateof.ai/) 📋                                                                                                                 |
| Technology Radar Volume 31 (October 2024)                                                | [thoughtworks's report](https://www.thoughtworks.com/radar) 📋                                                                                                |
| LlamaIndex Webinar: ColPali - Efficient Document Retrieval with Vision Language Models   | [LlamaIndex's Youtube video](https://youtu.be/nzcBvba7mzI?si=WL9MsyiAFJMyEolz) 📹                                                                             |
| PDF Retrieval with Vision Language Models                                                | [Jo's blog post #1 (Vespa)](https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/) 📝                                                          |
| Scaling ColPali to billions of PDFs with Vespa                                           | [Jo's blog post #2 (Vespa)](https://blog.vespa.ai/scaling-colpali-to-billions/) 📝                                                                            |
| Neural Search Talks: ColPali (with Manuel Faysse)                                        | [Zeta Alpha's Podcast](https://open.spotify.com/episode/2s6ljhd6VQTL2mIU9cFzCb) 📹                                                                            |
| Multimodal Document RAG with Llama 3.2 Vision and ColQwen2                               | [Zain's blog post (Together AI)](https://www.together.ai/blog/multimodal-document-rag-with-llama-3-2-vision-and-colqwen2) 📝                                  |
| ColPali: Document Retrieval with Vision Language Models                                  | [Antaripa Saha](https://antaripasaha.notion.site/ColPali-Efficient-Document-Retrieval-with-Vision-Language-Models-10f5314a5639803d94d0d7ac191bb5b1) 📝        |
| Minimalist diagrams explaining ColPali                                                   | [Leonie's ColPali diagrams on X](https://twitter.com/helloiamleonie/status/1839321865195851859)📝                                                            |
| Multimodal RAG with ColPali and Gemini : Financial Report Analysis Application           | [Jaykumaran's blog post (LearnOpenCV)](https://learnopencv.com/multimodal-rag-with-colpali/) 📝                                                               |
| Implement Multimodal RAG with ColPali and Vision Language Model Groq(Llava) and Qwen2-VL | [Plaban's blog post](https://medium.com/the-ai-forum/implement-multimodal-rag-with-colpali-and-vision-language-model-groq-llava-and-qwen2-vl-5c113b8c08fd) 📝 |
| multimodal AI. open-source. in a nutshell.                                               | [Merve's Youtube video](https://youtu.be/IoGaGfU1CIg?si=yEhxMqJYxvMzGyUm) 📹                                                                                  |
| Remove Complexity from Your RAG Applications                                             | [Kyryl's blog post (KOML)](https://kyrylai.com/2024/09/09/remove-complexity-from-your-rag-applications/) 📝                                                   |
| Late interaction & efficient Multi-modal retrievers need more than a vector index        | [Ayush Chaurasia (LanceDB)](https://blog.lancedb.com/late-interaction-efficient-multi-modal-retrievers-need-more-than-just-a-vector-index/) 📝                |
| Optimizing Document Retrieval with ColPali and Qdrant's Binary Quantization              | [Sabrina Aquino (Qdrant)]( https://youtu.be/_A90A-grwIc?si=MS5RV17D6sgirCRm)  📹                                                                              |
| Hands-On Multimodal Retrieval and Interpretability (ColQwen + Vespa)                     | [Antaripa Saha](https://www.analyticsvidhya.com/blog/2024/10/multimodal-retrieval-with-colqwen-vespa/) 📝                                                     |

</details>

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

@misc{macé2025vidorebenchmarkv2raising,
      title={ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval}, 
      author={Quentin Macé and António Loison and Manuel Faysse},
      year={2025},
      eprint={2505.17166},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.17166}, 
}
```
