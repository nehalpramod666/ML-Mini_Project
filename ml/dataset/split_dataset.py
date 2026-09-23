import json
import random

random.seed(42)

def split(in_path: str, train_path: str, val_path: str, val_frac: float = 0.1):
    rows = [json.loads(l) for l in open(in_path)]
    random.shuffle(rows)
    n_val = int(len(rows) * val_frac)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    
    print(f"train: {len(train_rows)} rows, val: {len(val_rows)} rows")

if __name__ == "__main__":
    split("ml/dataset/train_500.jsonl", "ml/dataset/train.jsonl", "ml/dataset/val.jsonl")