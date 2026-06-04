import os
from typing import List, Tuple, cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from PIL import Image

from colpali_engine.data.dataset import ColPaliEngineDataset, Corpus

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"


def load_train_set() -> ColPaliEngineDataset:
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    dataset = load_dataset(base_path + "colpali_train_set", split="train")

    train_dataset = ColPaliEngineDataset(dataset, pos_target_column_name="image")

    return train_dataset


def load_eval_set(dataset_path) -> ColPaliEngineDataset:
    dataset = load_dataset(dataset_path, split="test")

    return dataset


def load_train_set_ir(num_negs=0) -> ColPaliEngineDataset:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "manu/"
    corpus_data = load_dataset(base_path + "colpali-corpus", split="train")
    corpus = Corpus(corpus_data=corpus_data, doc_column_name="image")

    dataset = load_dataset(base_path + "colpali-queries", split="train")

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    if num_negs > 0:
        # keep only top 5 negative passages
        dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:num_negs]})
    print("Dataset size after filtering:", len(dataset))

    train_dataset = ColPaliEngineDataset(
        data=dataset,
        corpus=corpus,
        pos_target_column_name="positive_passages",
        neg_target_column_name="negative_passages" if num_negs else None,
    )

    return train_dataset


def load_train_set_detailed() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_train_set_with_tabfquad() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docmatix_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "docmatix-ir", split="train"))
    # dataset = dataset.select(range(100500))

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "Docmatix", "images", split="train"))

    return ds_dict, anchor_ds, "docmatix"


def load_wikiss() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "wiki-ss-nq", data_files="train.jsonl", split="train"))
    # dataset = dataset.select(range(400500))
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "wiki-ss-corpus", split="train"))

    return ds_dict, anchor_ds, "wikiss"


def load_train_set_with_docmatix() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
        "Docmatix_filtered_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot: List[Dataset] = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = concatenate_datasets(ds_tot)
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docvqa_dataset() -> DatasetDict:
    if USE_LOCAL_DATASET:
        dataset_doc = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="test"))
    else:
        dataset_doc = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation"))
        dataset_doc_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="test"))
        dataset_info = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation"))
        dataset_info_eval = cast(Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test"))

    # concatenate the two datasets
    dataset = concatenate_datasets([dataset_doc, dataset_info])
    dataset_eval = concatenate_datasets([dataset_doc_eval, dataset_info_eval])
    # sample 100 from eval dataset
    dataset_eval = dataset_eval.shuffle(seed=42).select(range(200))

    # rename question as query
    dataset = dataset.rename_column("question", "query")
    dataset_eval = dataset_eval.rename_column("question", "query")

    # create new column image_filename that corresponds to ucsf_document_id if not None, else image_url
    dataset = dataset.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )
    dataset_eval = dataset_eval.map(
        lambda x: {"image_filename": x["ucsf_document_id"] if x["ucsf_document_id"] is not None else x["image_url"]}
    )

    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    return ds_dict


def load_dummy_dataset() -> List[DatasetDict]:
    # create a dataset from the queries and images
    queries_1 = ["What is the capital of France?", "What is the capital of Germany?"]
    queries_2 = ["What is the capital of Italy?", "What is the capital of Spain?"]

    images_1 = [Image.new("RGB", (100, 100)) for _ in range(2)]
    images_2 = [Image.new("RGB", (120, 120)) for _ in range(2)]

    dataset_1 = Dataset.from_list([{"query": q, "image": i} for q, i in zip(queries_1, images_1)])
    dataset_2 = Dataset.from_list([{"query": q, "image": i} for q, i in zip(queries_2, images_2)])

    return DatasetDict(
        {
            "train": DatasetDict({"dataset_1": dataset_1, "dataset_2": dataset_2}),
            "test": DatasetDict({"dataset_1": dataset_2, "dataset_2": dataset_1}),
        }
    )


def load_multi_qa_datasets() -> List[DatasetDict]:
    dataset_args = [
        ("vidore/colpali_train_set"),
        ("llamaindex/vdr-multilingual-train", "de"),
        ("llamaindex/vdr-multilingual-train", "en"),
        ("llamaindex/vdr-multilingual-train", "es"),
        ("llamaindex/vdr-multilingual-train", "fr"),
        ("llamaindex/vdr-multilingual-train", "it"),
    ]

    train_datasets = {}
    test_datasets = {}
    for args in dataset_args:
        dataset_name = args[0] + "_" + args[1]
        dataset = load_dataset(*args)
        if "test" in dataset:
            train_datasets[dataset_name] = dataset["train"]
            test_datasets[dataset_name] = dataset["test"]
        else:
            train_dataset, test_dataset = dataset.split_by_ratio(test_size=200)
            train_datasets[dataset_name] = train_dataset
            test_datasets[dataset_name] = test_dataset

    return DatasetDict({"train": DatasetDict(train_datasets), "test": DatasetDict(test_datasets)})


class TestSetFactory:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __call__(self, *args, **kwargs):
        dataset = load_dataset(self.dataset_path, split="test")
        return dataset


if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)
