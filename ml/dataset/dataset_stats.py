import json
from collections import Counter

def analyze(path: str):
    rows = [json.loads(l) for l in open(path)]
    metrics_list = [json.loads(r["input"]) for r in rows]

    state_counts = Counter(m["state"] for m in metrics_list)
    defect_counts = Counter(m["dominant_defect"] for m in metrics_list)

    print(f"Total rows: {len(rows)}")
    print(f"State distribution: {dict(state_counts)}")
    print(f"Defect distribution: {dict(defect_counts)}")

    for field in ["blur", "contrast", "noise"]:
        vals = [m[field] for m in metrics_list]
        print(f"{field}: min={min(vals):.1f}, max={max(vals):.1f}, "
              f"mean={sum(vals)/len(vals):.1f}")

if __name__ == "__main__":
    print("===Train===")
    analyze("ml/dataset/train.jsonl")
    print("=== Val ===")
    analyze("ml/dataset/val.jsonl")