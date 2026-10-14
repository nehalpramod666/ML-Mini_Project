import difflib
import json
import re
from pathlib import Path

from ml import llm_engine as e


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = PROJECT_ROOT / "ml" / "dataset" / "train_chat.jsonl"
VAL_FILE = PROJECT_ROOT / "ml" / "dataset" / "val.jsonl"
SAMPLE_LIMIT = 100


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_messages(row):
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list):
        raise ValueError("Training row has no messages/conversations list")
    return messages


def get_training_system_prompt():
    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    first_row = read_jsonl(TRAIN_FILE)[0]
    messages = get_messages(first_row)

    for message in messages:
        if message.get("role") == "system":
            return message["content"]

    raise ValueError("No system message found in the first training row")


def legacy_prompt(flat, system_prompt):
    """Reproduce the previous Day 11 hand-built prompt."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{json.dumps(flat)}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


NUM = re.compile(
    r"(\d+)\s*/\s*100|(?:blur|contrast|noise)(?: score)?(?: of)?\s+(\d+)",
    re.IGNORECASE,
)


def numeric_ok(text, flat):
    """Check that numerical metric claims match values in the input."""
    allowed = {round(flat[key]) for key in ("blur", "contrast", "noise")}
    found_numbers = NUM.findall(text)
    return all(int(first or second) in allowed for first, second in found_numbers)


def main():
    train_sys = get_training_system_prompt()
    prompt_matches = e.SYSTEM_PROMPT == train_sys

    print("Engine prompt == training prompt:", prompt_matches)

    if not prompt_matches:
        print("TRAINING:", repr(train_sys))
        print("ENGINE:  ", repr(e.SYSTEM_PROMPT))

    # Ensure both inference paths use the exact training system text.
    # This changes only the current process, not the source file.
    e.SYSTEM_PROMPT = train_sys

    rows = read_jsonl(VAL_FILE)[:SAMPLE_LIMIT]
    if not rows:
        raise ValueError(f"No validation rows found in {VAL_FILE}")

    llm = e.get_llm()
    metadata = getattr(llm, "metadata", {}) or {}

    print(
        "GGUF has chat template:",
        "tokenizer.chat_template" in metadata,
    )
    print("Validation rows:", len(rows))

    stats = {
        "chat": {"numeric_ok": 0, "similarity": 0.0},
        "legacy": {"numeric_ok": 0, "similarity": 0.0},
    }

    for index, row in enumerate(rows, start=1):
        raw_input = row["input"]
        flat_input = (
            json.loads(raw_input)
            if isinstance(raw_input, str)
            else raw_input
        )

        flat = e.to_llm_input(flat_input)
        target = row["output"]

        # New serving path: GGUF chat template.
        new_output = e.generate_report(flat)

        # Legacy path: same prompt text, hand-built formatting.
        with e._gen_lock:
            response = llm(
                legacy_prompt(flat, train_sys),
                max_tokens=e.MAX_TOKENS,
                temperature=0.0,
                stop=["<|eot_id|>"],
            )

        old_output = response["choices"][0]["text"].strip()

        for name, output in (
            ("chat", new_output),
            ("legacy", old_output),
        ):
            stats[name]["numeric_ok"] += int(numeric_ok(output, flat))
            stats[name]["similarity"] += difflib.SequenceMatcher(
                None, output, target
            ).ratio()

        if new_output != old_output and index <= 3:
            print(f"\n--- Difference on row {index} ---")
            print("CHAT:", repr(new_output))
            print("LEGACY:", repr(old_output))

        if index % 10 == 0:
            print(f"Compared {index}/{len(rows)} rows")

    n = len(rows)
    print("\n--- Controlled A/B results ---")

    for name, values in stats.items():
        print(
            f"{name}: numeric fidelity "
            f"{values['numeric_ok']}/{n}, "
            f"mean similarity to target "
            f"{values['similarity'] / n:.3f}"
        )


if __name__ == "__main__":
    main()