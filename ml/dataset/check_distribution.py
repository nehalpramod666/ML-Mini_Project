import json
from collections import Counter

with open(
    "ml/dataset/batch_50.jsonl",
    encoding="utf-8"
) as f:
    rows = [json.loads(line) for line in f]

metrics = [
    json.loads(row["input"])
    for row in rows
]

states = Counter(
    item["state"]
    for item in metrics
)

defects = Counter(
    item["dominant_defect"]
    for item in metrics
)

print("States:")
for state, count in states.items():
    print(f"  {state}: {count}")

print("\nDominant defects:")
for defect, count in defects.items():
    print(f"  {defect}: {count}")