import os
from typing import List, Tuple, cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"


def add_metadata_column(dataset, column_name, value):
    def add_source(example):
        example[column_name] = value
        return example

    return dataset.map(add_source)


def load_train_set() -> DatasetDict:
    ds_path = "colpali_train_set"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_dict = cast(DatasetDict, load_dataset(base_path + ds_path))
    return ds_dict


def load_clean_colpali_dataset() -> DatasetDict:
    ds_name = "clean-colpali-dataset"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "antonioloison/"
    ds_path = base_path + ds_name
    corpus = cast(DatasetDict, load_dataset(ds_path, "corpus"))["train"]
    queries = cast(DatasetDict, load_dataset(ds_path, "queries"))

    new_queries = DatasetDict({"train": queries["train"], "test": queries["test"]})

    return new_queries, corpus, "beir"


def load_clean_colpali_dataset_dedup() -> DatasetDict:
    ds_name = "clean-colpali-dataset"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "antonioloison/"
    ds_path = base_path + ds_name
    corpus = cast(DatasetDict, load_dataset(ds_path, "corpus"))["train"]
    queries = cast(DatasetDict, load_dataset(ds_path, "queries"))

    new_queries = DatasetDict({"train": queries["train"], "test": queries["test"]})

    return new_queries, corpus, "beir-dedup"


def load_mixed_dataset() -> DatasetDict:
    ds_name = "colpali-mixed-dataset"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "antonioloison/"
    ds_path = base_path + ds_name
    corpus = cast(DatasetDict, load_dataset(ds_path, "corpus"))["train"]
    queries = cast(DatasetDict, load_dataset(ds_path, "queries"))

    new_queries = DatasetDict({"train": queries["train"], "test": queries["test"]})

    return new_queries, corpus, "beir"


def load_mixed_fr_en_dataset() -> DatasetDict:
    ds_name = "colpali-mixed-dataset-fr-en"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "antonioloison/"
    ds_path = base_path + ds_name
    corpus = cast(DatasetDict, load_dataset(ds_path, "corpus"))["train"]
    queries = cast(DatasetDict, load_dataset(ds_path, "queries"))

    new_queries = DatasetDict({"train": queries["train"], "test": queries["test"]})

    return new_queries, corpus, "beir"


def load_mixed_fr_en_dataset_with_translations() -> DatasetDict:
    ds_name = "colpali-mixed-dataset-fr-en"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "antonioloison/"
    ds_path = base_path + ds_name
    corpus = cast(DatasetDict, load_dataset(ds_path, "corpus"))["train"]
    queries = cast(DatasetDict, load_dataset(ds_path, "queries"))

    new_queries = DatasetDict({"train": queries["train"], "test": queries["test"]})

    return new_queries, corpus, "beir-with-translations"


def load_vdsid_train_set(ds_name: str) -> DatasetDict:
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_path = base_path + ds_name
    corpus = cast(DatasetDict, load_dataset(ds_path, "corpus"))
    queries = cast(DatasetDict, load_dataset(ds_path, "queries"))
    qrels = cast(DatasetDict, load_dataset(ds_path, "qrels"))

    corpus_train_length = len(corpus["train"])

    def add_corpus_id(example):
        example["corpus_id"] = example["corpus_id"] + corpus_train_length
        return example

    corpus["test"] = corpus["test"].map(add_corpus_id)
    new_corpus = concatenate_datasets([corpus["train"], corpus["test"]])

    train_qrels_df = qrels["train"].to_pandas()
    test_qrels_df = qrels["test"].to_pandas()

    def add_corpus_indexes_train(example):
        disable_progress_bar()
        sub_qrels = train_qrels_df[train_qrels_df["query_id"] == example["query_id"]]
        enable_progress_bar()
        positive_docs = [(x.corpus_id, x.score) for x in sub_qrels.itertuples()]
        example["positive_docs"] = positive_docs
        return example

    def add_corpus_indexes_test(example):
        disable_progress_bar()
        sub_qrels = test_qrels_df[test_qrels_df["query_id"] == example["query_id"]]
        enable_progress_bar()
        positive_docs = [(x.corpus_id + corpus_train_length, x.score) for x in sub_qrels.itertuples()]
        example["positive_docs"] = positive_docs
        return example

    queries["train"] = queries["train"].map(add_corpus_indexes_train)
    queries["test"] = queries["test"].map(add_corpus_indexes_test)

    new_queries = DatasetDict({"train": queries["train"], "test": queries["test"]})

    return new_queries, new_corpus, "beir"


def load_vdsid_train_set_big() -> DatasetDict:
    ds_name = "vdsid_beir_full_positive"
    return load_vdsid_train_set(ds_name)


def load_vdsid_train_set_small() -> DatasetDict:
    ds_name = "vdsid_beir_full_positive_small"
    return load_vdsid_train_set(ds_name)


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


def load_train_set_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "manu/"
    dataset = cast(Dataset, load_dataset(base_path + "colpali-queries", split="train"))

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    print("Dataset size after filtering:", len(dataset))

    # keep only top 50 negative passages
    dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:50]})

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    anchor_ds = cast(Dataset, load_dataset(base_path + "colpali-corpus", split="train"))
    return ds_dict, anchor_ds, "vidore"


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


class TestSetFactory:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __call__(self, *args, **kwargs):
        dataset = load_dataset(self.dataset_path, split="test")
        return dataset


if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)
