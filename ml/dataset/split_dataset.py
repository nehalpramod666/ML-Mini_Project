import json
import random
from pathlib import Path


INPUT_FILE = Path("ml/dataset/train_5000_raw.jsonl")
TRAIN_FILE = Path("ml/dataset/train.jsonl")
VAL_FILE = Path("ml/dataset/val.jsonl")

VAL_RATIO = 0.10
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split(
    in_path: str,
    train_path: str,
    val_path: str,
    val_frac: float = 0.1,
) -> None:
    input_path = Path(in_path)
    train_output = Path(train_path)
    val_output = Path(val_path)

    rows = load_jsonl(input_path)

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    random.seed(SEED)
    random.shuffle(rows)

    n_val = int(len(rows) * val_frac)

    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    write_jsonl(train_output, train_rows)
    write_jsonl(val_output, val_rows)

    print(f"Total: {len(rows)}")
    print(f"Train: {len(train_rows)}")
    print(f"Val: {len(val_rows)}")


if __name__ == "__main__":
    split(
        str(INPUT_FILE),
        str(TRAIN_FILE),
        str(VAL_FILE),
    )