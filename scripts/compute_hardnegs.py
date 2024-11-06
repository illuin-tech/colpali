from typing import cast

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import BiQwen2, BiQwen2Processor
from colpali_engine.utils.dataset_transformation import load_train_set

train_set = load_train_set()


COMPUTE_EMBEDDINGS = False
COMPUTE_HARDNEGS = False

if COMPUTE_HARDNEGS or COMPUTE_EMBEDDINGS:
    print("Loading base model")
    model = BiQwen2.from_pretrained(
        "./models/biqwen2-warmup-256-newpad-0e",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    ).eval()

    print("Loading processor")
    processor = BiQwen2Processor.from_pretrained("./models/biqwen2-warmup-256-newpad-0e")

if COMPUTE_EMBEDDINGS:
    print("Loading images")
    print("Images loaded")

    document_set = train_set["train"]
    print("Filtering dataset")
    print(document_set)
    initial_list = document_set["image_filename"]
    _, unique_indices = np.unique(initial_list, return_index=True, axis=0)
    filtered_dataset = document_set.select(unique_indices.tolist())
    filtered_dataset = filtered_dataset.map(
        lambda example: {"image": example["image"], "image_filename": example["image_filename"]}, num_proc=16
    )
    # keep only column image and image_filename and source if it exists
    cols_to_remove = [col for col in filtered_dataset.column_names if col not in ["image", "image_filename"]]
    filtered_dataset = filtered_dataset.remove_columns(cols_to_remove)
    # save it
    print("Saving filtered dataset")
    print(filtered_dataset)
    filtered_dataset.save_to_disk("data_dir/filtered_dataset", max_shard_size="200MB")

    print("Processing images")
    # run inference - docs
    dataloader = DataLoader(
        filtered_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: processor.process_images([a["image"] for a in x]),
    )
    print("Computing embeddings")

    ds = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    ds = torch.stack(ds)

    # save embeddings
    torch.save(ds, "data_dir/filtered_dataset_embeddings.pt")

if not COMPUTE_EMBEDDINGS:
    ds = torch.load("data_dir/filtered_dataset_embeddings.pt")


if COMPUTE_HARDNEGS:
    # compute hard negatives
    ds = cast(torch.Tensor, ds).to("cuda")

    # iterate on the train set
    mined_hardnegs = []

    for i in tqdm(range(0, len(train_set["train"]), 8)):
        samples = train_set["train"][i : i + 8]
        batch_query = processor.process_queries(samples["query"])
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)

        # compute scores
        scores = torch.einsum("bd,cd->bc", embeddings_query, ds)
        # get top 100 indexes
        top100 = scores.topk(100, dim=1).indices
        # indices to list
        top100 = top100.tolist()
        # append to mined_hardnegs
        mined_hardnegs.extend(top100)

    # save mined hardnegs as txt
    with open("data_dir/mined_hardnegs_filtered.txt", "w") as f:
        for item in mined_hardnegs:
            f.write("%s\n" % item)


with open("data_dir/mined_hardnegs_filtered.txt") as f:
    mined_hardnegs = f.readlines()


filtered_dataset = datasets.load_from_disk("data_dir/filtered_dataset")
filenames = list(filtered_dataset["image_filename"])


def mapper_fn(example, idx):
    tmp = {
        "negative_passages": [int(x) for x in mined_hardnegs[idx][1:-2].strip().split(",")],
        "query": example["query"],
        "positive_passages": [filenames.index(example["image_filename"])],
    }

    tmp["gold_in_top_100"] = tmp["positive_passages"][0] in tmp["negative_passages"]
    # remove gold index from negs if it is there
    if tmp["gold_in_top_100"]:
        tmp["negative_passages"].remove(tmp["positive_passages"][0])
    return tmp


final_dataset = train_set["train"].map(mapper_fn, with_indices=True, num_proc=16)
# drop image
final_dataset = final_dataset.remove_columns("image")
final_dataset.save_to_disk("data_dir/final_dataset")
