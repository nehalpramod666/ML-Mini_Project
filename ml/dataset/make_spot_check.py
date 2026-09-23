import json
import random

INPUT_PATH = "ml/dataset/train_500.jsonl"
OUTPUT_PATH = "ml/dataset/spot_check_sample.jsonl"

with open(INPUT_PATH, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

sample = random.sample(rows, 50)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in sample:
        f.write(json.dumps(row) + "\n")

print("Created 50-row spot-check sample.")