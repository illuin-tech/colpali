import argparse
import os

from datasets import load_dataset
from openai import OpenAI
from pqdm.processes import pqdm

client = OpenAI()


def reformulate_with_4o(query):
    query_template = """
    Given a query:
    1. Repeat the query.
    2. Identify the essential problem.
    3. Think step by step to reason and describe what information could be relevant and helpful to address
    the questions in detail.
    4. Draft an answer with as many thoughts as you have.

    Answer in the same language as the query.
    Query: {query}
    """
    prompt = query_template.format(query=query)
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()


def process_dataset(dataset_name, query_column="query"):
    """
    Download dataset, reformulate queries using 4O, and reupload.
    Args:
        dataset_name: Name of the HuggingFace dataset
        query_column: Column containing queries
    """

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, "queries", split="test")

    # Import pqdm for parallel processing

    # Determine the number of cores to use
    n_jobs = os.cpu_count()
    print(f"Using {n_jobs} processes for parallel processing")

    # Prepare the dataset for processing
    queries = dataset[query_column]
    print(f"Processing {len(queries)} queries using 4O reformulation...")

    # Process queries in parallel
    reformulated_queries = pqdm(list(queries), reformulate_with_4o, n_jobs=n_jobs)

    print("Reformulation complete. Adding to dataset...")
    print(reformulated_queries[:5])  # Print first 5 reformulated queries for verification

    # Add the reformulated queries as a new column
    updated_dataset = dataset.add_column("gpt-4o-reasoning", reformulated_queries)
    print("Reformulation complete!")

    # Push to the Hugging Face Hub if auth_token is provided
    updated_dataset.push_to_hub(dataset_name, "queries", split="test")
    print("Upload complete!")

    return updated_dataset


def main():
    parser = argparse.ArgumentParser(description="Reformulate queries in a dataset using 4O technique")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--query_column", default="query", help="Column containing queries")
    args = parser.parse_args()

    process_dataset(args.dataset, args.query_column)


if __name__ == "__main__":
    main()
