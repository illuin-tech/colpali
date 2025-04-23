import os
from typing import List, Literal, Tuple, cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from colpali_engine.data.corpus import LocalCorpus, MapCorpus
from colpali_engine.data.dataset import IRColumn, IRDataset

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


def load_train_set_ir() -> IRDataset:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    corpus = MapCorpus.from_dataset(
        corpus_data=load_dataset("manu/colpali-corpus", split="train"),
        docid_column=None,
    )

    dataset = load_dataset("manu/colpali-queries", split="train")

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    # keep only top 5 negative passages
    print("Dataset size after filtering:", len(dataset))

    train_dataset = IRDataset(
        data=dataset,
        query_column="query",
        pos_target_column=IRColumn("positive_passages", corpus_column="image"),
        corpus=corpus,
    )

    return train_dataset


def load_train_set_ir_negs() -> IRDataset:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    corpus = MapCorpus.from_dataset(
        corpus_data=load_dataset("manu/colpali-corpus", split="train"),
        docid_column=None,
    )

    dataset = load_dataset("manu/colpali-queries", split="train")

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    # keep only top 50 negative passages
    dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:5]})
    print("Dataset size after filtering:", len(dataset))

    train_dataset = IRDataset(
        data=dataset,
        query_column="query",
        pos_target_column=IRColumn("positive_passages", corpus_column="image"),
        neg_target_column=IRColumn("negative_passages", corpus_column="image"),
        corpus=corpus,
    )

    return train_dataset


def load_mmeb(subset: str, corpus_path: str, type: Literal["t2i", "i2t"], use_negatives: bool = False) -> IRDataset:
    corpus = LocalCorpus(corpus_path=corpus_path)
    dataset = load_dataset("TIGER-Lab/MMEB-train", subset, split="original")

    if type == "t2i":
        query_col = "qry"
        pos_col = IRColumn("pos_image_path", corpus_column="doc")
        neg_col = IRColumn("neg_image_path", corpus_column="doc") if use_negatives else None
    elif type == "i2t":
        query_col = IRColumn("qry_image_path", corpus_column="doc")
        pos_col = "pos_text"
        neg_col = "neg_text" if use_negatives else None
    else:
        raise ValueError("type must be either 't2i' or 'i2t'")

    return IRDataset(
        data=dataset,
        query_column=query_col,
        pos_target_column=pos_col,
        neg_target_column=neg_col,
        corpus=corpus,
    )


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


class TestSetFactory:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __call__(self, *args, **kwargs):
        dataset = load_dataset(self.dataset_path, split="test")
        return dataset


if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)
