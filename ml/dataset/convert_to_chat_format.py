import json

SYSTEM_PROMPT = (
    "You are a clinical imaging quality assistant. Given DICOM quality metrics "
    "(blur, contrast, noise scores, overall state, and dominant defect), explain "
    "the defect and give a concise, actionable retake instruction."
)

def convert_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
    }

def convert_file(in_path: str, out_path: str):
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            fout.write(json.dumps(convert_row(row)) + "\n")
    print(f"Converted {in_path} -> {out_path}")

if __name__ == "__main__":
    convert_file("ml/dataset/train.jsonl", "ml/dataset/train_chat.jsonl")
    convert_file("ml/dataset/val.jsonl", "ml/dataset/val_chat.jsonl")