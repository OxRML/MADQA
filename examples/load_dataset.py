"""
Load and inspect the MADQA dataset.

Usage:
    pip install datasets
    python load_dataset.py
"""

from collections import Counter
from datasets import load_dataset

DATASET_NAME = "OxRML/MADQA"

dataset = load_dataset(DATASET_NAME)
print(f"Splits: {list(dataset.keys())}")
for split_name, split_data in dataset.items():
    print(f"  {split_name}: {len(split_data)} examples")

print(f"\nColumns: {dataset['dev'].column_names}")

# Show a sample
example = dataset["dev"][0]
print(f"\nSample question:")
print(f"  ID:       {example['id']}")
print(f"  Question: {example['question']}")
print(f"  Answers:  {example['answer_variants']}")
print(f"  Evidence: {example['evidence']}")
print(f"  Category: {example['document_category']}")
print(f"  Domain:   {example['domain']}")

# Basic statistics
dev = dataset["dev"]
categories = Counter(ex["document_category"] for ex in dev)
domains = Counter(ex["domain"] for ex in dev)

print(f"\nDev set statistics:")
print(f"  Questions: {len(dev)}")
print(f"  Categories ({len(categories)}): {categories.most_common(5)} ...")
print(f"  Domains ({len(domains)}):    {domains.most_common(5)} ...")

evidence_counts = [len(ex["evidence"]) for ex in dev]
print(f"  Evidence pages per question: min={min(evidence_counts)}, "
      f"max={max(evidence_counts)}, avg={sum(evidence_counts)/len(evidence_counts):.1f}")
