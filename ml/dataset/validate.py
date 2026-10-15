import json


VALID_STATES = {
    "ACCEPT",
    "BORDERLINE",
    "REJECT",
}

VALID_DEFECTS = {
    "motion_blur",
    "low_contrast",
    "high_contrast",
    "sensor_noise",
    "none",
}


def validate_dataset(path: str, expected_rows: int = 500) -> None:
    bad = []
    count = 0

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            try:
                row = json.loads(line)
                metrics = json.loads(row["input"])
            except (json.JSONDecodeError, KeyError) as exc:
                bad.append(
                    (line_number, f"Invalid JSON/structure: {exc}")
                )
                continue

            count += 1

            # -------------------------------------------------
            # 1. Check outer row keys
            # -------------------------------------------------
            required_row_keys = {
                "instruction",
                "input",
                "output",
            }

            if set(row.keys()) != required_row_keys:
                bad.append(
                    (
                        line_number,
                        f"Invalid row keys: {set(row.keys())}",
                    )
                )

            # -------------------------------------------------
            # 2. Check metric keys
            # -------------------------------------------------
            required_metric_keys = {
                "blur",
                "contrast",
                "noise",
                "state",
                "dominant_defect",
            }

            if set(metrics.keys()) != required_metric_keys:
                bad.append(
                    (
                        line_number,
                        f"Invalid metric keys: {set(metrics.keys())}",
                    )
                )
                continue

            # -------------------------------------------------
            # 3. Check state and defect values
            # -------------------------------------------------
            state = metrics["state"]
            defect = metrics["dominant_defect"]

            if state not in VALID_STATES:
                bad.append(
                    (
                        line_number,
                        f"Invalid state: {state}",
                    )
                )

            if defect not in VALID_DEFECTS:
                bad.append(
                    (
                        line_number,
                        f"Invalid dominant_defect: {defect}",
                    )
                )

            # -------------------------------------------------
            # 4. Check metric types and basic range
            # -------------------------------------------------
            for metric_name in ["blur", "contrast", "noise"]:
                value = metrics[metric_name]

                if not isinstance(value, (int, float)):
                    bad.append(
                        (
                            line_number,
                            f"{metric_name} is not numeric: {value}",
                        )
                    )
                elif not 0 <= value <= 100:
                    bad.append(
                        (
                            line_number,
                            f"{metric_name} out of range: {value}",
                        )
                    )

            # -------------------------------------------------
            # 5. State ↔ dominant defect consistency
            # -------------------------------------------------
            if state == "ACCEPT" and defect != "none":
                bad.append(
                    (
                        line_number,
                        "ACCEPT row has non-none dominant defect",
                    )
                )

            if state != "ACCEPT" and defect == "none":
                bad.append(
                    (
                        line_number,
                        "Non-ACCEPT row has dominant_defect=none",
                    )
                )

            # -------------------------------------------------
            # 6. Dominant defect ↔ metric range consistency
            # -------------------------------------------------

            if defect == "motion_blur":

                if state == "REJECT":
                    if not 65 <= metrics["blur"] <= 100:
                        bad.append(
                            (
                                line_number,
                                (
                                    "REJECT motion_blur has invalid "
                                    f"blur range: {metrics['blur']}"
                                ),
                            )
                        )

                elif state == "BORDERLINE":
                    if not 40 <= metrics["blur"] <= 64:
                        bad.append(
                            (
                                line_number,
                                (
                                    "BORDERLINE motion_blur has invalid "
                                    f"blur range: {metrics['blur']}"
                                ),
                            )
                        )

            elif defect == "low_contrast":

                if state == "REJECT":
                    if not 0 <= metrics["contrast"] <= 35:
                        bad.append(
                            (
                                line_number,
                                (
                                    "REJECT low_contrast has invalid "
                                    f"contrast range: {metrics['contrast']}"
                                ),
                            )
                        )

                elif state == "BORDERLINE":
                    if not 35 <= metrics["contrast"] <= 49:
                        bad.append(
                            (
                                line_number,
                                (
                                    "BORDERLINE low_contrast has invalid "
                                    f"contrast range: {metrics['contrast']}"
                                ),
                            )
                        )

            elif defect == "high_contrast":

                if state == "REJECT":
                    if not 65 <= metrics["contrast"] <= 100:
                        bad.append(
                            (
                                line_number,
                                (
                                    "REJECT high_contrast has invalid "
                                    f"contrast range: {metrics['contrast']}"
                                ),
                            )
                        )

                elif state == "BORDERLINE":
                    if not 60 <= metrics["contrast"] <= 64:
                        bad.append(
                            (
                                line_number,
                                (
                                    "BORDERLINE high_contrast has invalid "
                                    f"contrast range: {metrics['contrast']}"
                                ),
                            )
                        )

            elif defect == "sensor_noise":

                if state == "REJECT":
                    if not 65 <= metrics["noise"] <= 100:
                        bad.append(
                            (
                                line_number,
                                (
                                    "REJECT sensor_noise has invalid "
                                    f"noise range: {metrics['noise']}"
                                ),
                            )
                        )

                elif state == "BORDERLINE":
                    if not 40 <= metrics["noise"] <= 64:
                        bad.append(
                            (
                                line_number,
                                (
                                    "BORDERLINE sensor_noise has invalid "
                                    f"noise range: {metrics['noise']}"
                                ),
                            )
                        )

            # -------------------------------------------------
            # 7. P10 — Cross-metric argmax invariant (DEFERRED)
            #    A cross-metric argmax check (e.g. assert blur > noise when
            #    dominant_defect is motion_blur) would require all three
            #    metrics to be on a common, comparable severity scale.
            #    In the current training schema:
            #      blur    is a synthetic score derived from Laplacian variance
            #      contrast is a synthetic directional score
            #      noise   is a synthetic score derived from an SNR proxy
            #    The generator places the dominant defect's metric in a
            #    state-dependent high range and the others in a neutral band.
            #    Sections 5 and 6 already verify per-field ranges for each
            #    declared defect, which is the strongest invariant the current
            #    schema supports.
            #    Cross-metric argmax will be added once D7(b) confirms a
            #    shared severity representation for all three fields.
            #    Tracking issue: P10, blocked on D7(b).
            # -------------------------------------------------

    # ---------------------------------------------------------
    # 7. Check total number of rows
    # ---------------------------------------------------------
    print(f"Rows checked: {count}")

    if count != expected_rows:
        bad.append(
            (
                0,
                f"Expected {expected_rows} rows, found {count}",
            )
        )

    print(f"Inconsistent rows: {len(bad)}")

    # ---------------------------------------------------------
    # 8. Report problems
    # ---------------------------------------------------------
    if bad:
        print("\nProblems:")

        for line_number, message in bad[:20]:
            if line_number:
                print(f"  Line {line_number}: {message}")
            else:
                print(f"  {message}")

        if len(bad) > 20:
            print(f"\n  ...and {len(bad) - 20} more problems.")

        raise SystemExit(1)

    print(f"0 inconsistent rows out of {count}")


if __name__ == "__main__":
    validate_dataset(
        "ml/dataset/train_5000_raw.jsonl",
        expected_rows=5000,
    )
