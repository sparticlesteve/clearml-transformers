"""Upload a Hugging Face text classification dataset as a raw ClearML Dataset
(train.csv/test.csv with "text" and "label" columns), for use as the
pipeline's raw_dataset_id."""
import argparse
import os

from datasets import load_dataset
from clearml import Dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hf-dataset", default="imdb",
                    help="Hugging Face dataset, e.g. imdb or cornell-movie-review-data/rotten_tomatoes")
parser.add_argument("--name", default=None, help="ClearML dataset name (default: <hf name>_raw)")
parser.add_argument("--project", default="SC26 demo-dev/LLM pipeline")
parser.add_argument("--tags", nargs="*", default=["sc26-test", "llm-pipeline"])
args = parser.parse_args()

dataset_name = args.name or args.hf_dataset.split("/")[-1] + "_raw"

output_folder = f"raw_{dataset_name}"
os.makedirs(output_folder, exist_ok=True)
train_path = os.path.join(output_folder, "train.csv")
test_path = os.path.join(output_folder, "test.csv")

hf_dataset = load_dataset(args.hf_dataset)
hf_dataset["train"].to_csv(train_path, index=False)
hf_dataset["test"].to_csv(test_path, index=False)

raw_dataset = Dataset.create(
    dataset_project=args.project,
    dataset_name=dataset_name,
    dataset_tags=args.tags,
)
raw_dataset.add_files(output_folder)

raw_dataset.upload()
raw_dataset.finalize()

print(f"Done. Dataset ID: {raw_dataset.id}")
