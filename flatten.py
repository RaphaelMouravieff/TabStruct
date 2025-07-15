from datasets import load_from_disk, DatasetDict
import os

# Load the HF dataset
dataset_path = "data/wikisql"
dataset = load_from_disk(dataset_path)

def flatten_answers(example):
    if isinstance(example["answers"], list):
        example["answers"] = ", ".join(example["answers"])
    return example

# Apply the transformation to all splits
flattened_dataset = DatasetDict()
for split in dataset.keys():
    flattened_dataset[split] = dataset[split].map(flatten_answers)

# Save to new directory
save_path = "data/wikisql_flattened"
os.makedirs(save_path, exist_ok=True)
flattened_dataset.save_to_disk(save_path)

print(f"✅ Flattened dataset saved to: {save_path}")


