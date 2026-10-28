import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "reports" / "stage1_baseline_20261027" / "per_image.csv"
OUTPUT = ROOT / "reports" / "day20_score_severity_20261028"

NUMERIC_FIELDS = [
    "fuzzy_score",
    "v2_fuzzy_score",
    "v2_blur_severity",
    "v2_contrast_severity",
    "v2_noise_severity",
]

SEVERITY_FIELDS = [
    "v2_blur_severity",
    "v2_contrast_severity",
    "v2_noise_severity",
]


def numeric(value):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return None


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    if denom == 0:
        return None
    return sum(a*b for a, b in zip(dx, dy)) / denom


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0

    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1

        # Average of the 1-based ranks for tied values.
        rank = ((position + 1) + end) / 2.0
        for j in range(position, end):
            ranks[order[j]] = rank
        position = end

    return ranks


def correlation_record(group_name, metric, records):
    pairs = []

    for row in records:
        score = numeric(row.get("fuzzy_score"))
        value = numeric(row.get(metric))
        if score is not None and value is not None:
            pairs.append((score, value))

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    return {
        "group": group_name,
        "score_field": "fuzzy_score",
        "severity_field": metric,
        "n_pairs": len(pairs),
        "pearson_r": pearson(xs, ys),
        "spearman_rho": (
            pearson(average_ranks(xs), average_ranks(ys))
            if len(pairs) >= 2 else None
        ),
        "interpretation_note": (
            "Descriptive association only; not evidence of causation "
            "or clinical validity."
        ),
    }


def distribution(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {
            "n": 0, "min": None, "median": None,
            "mean": None, "max": None
        }
    return {
        "n": len(clean),
        "min": min(clean),
        "median": statistics.median(clean),
        "mean": statistics.mean(clean),
        "max": max(clean),
    }


def main():
    if not INPUT.is_file():
        raise SystemExit(f"Baseline input missing: {INPUT}")

    if OUTPUT.exists():
        raise SystemExit(
            f"Output directory already exists; refusing to overwrite: {OUTPUT}"
        )

    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 100:
        raise SystemExit(
            f"Expected 100 existing baseline rows; found {len(rows)}."
        )

    required = {
        "injected_defect", "fuzzy_score", "fuzzy_state",
        "v2_blur_severity", "v2_contrast_severity",
        "v2_noise_severity", "v2_state", "v2_dominant_defect",
        "result_status",
    }

    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"Required CSV columns missing: {sorted(missing)}")

    analysis_rows = []

    for row in rows:
        out = {
            "filename": row.get("filename", ""),
            "injected_defect": row.get("injected_defect", ""),
            "result_status": row.get("result_status", ""),
            "fuzzy_score": numeric(row.get("fuzzy_score")),
            "fuzzy_state": row.get("fuzzy_state", ""),
            "v2_fuzzy_score": numeric(row.get("v2_fuzzy_score")),
            "v2_state": row.get("v2_state", ""),
            "v2_dominant_defect": row.get("v2_dominant_defect", ""),
        }

        severities = []
        for field in SEVERITY_FIELDS:
            value = numeric(row.get(field))
            out[field] = value
            if value is not None:
                severities.append(value)

        out["maximum_severity"] = max(severities) if severities else None
        out["sum_severity"] = sum(severities) if severities else None
        analysis_rows.append(out)

    successful = [
        row for row in analysis_rows
        if row["result_status"] == "success"
    ]

    # Overall and within-defect descriptive correlations.
    correlation_rows = []
    groups = [("overall", successful)]

    for defect in sorted({r["injected_defect"] for r in successful}):
        groups.append((
            defect,
            [r for r in successful if r["injected_defect"] == defect],
        ))

    for group_name, group_rows in groups:
        for metric in SEVERITY_FIELDS + ["maximum_severity", "sum_severity"]:
            correlation_rows.append(
                correlation_record(group_name, metric, group_rows)
            )

    score_distribution = distribution([
        r["fuzzy_score"] for r in successful
    ])

    score_by_defect = {}
    for defect in sorted({r["injected_defect"] for r in analysis_rows}):
        group = [r for r in analysis_rows if r["injected_defect"] == defect]
        score_by_defect[defect] = {
            "record_count": len(group),
            "score_distribution": distribution([
                r["fuzzy_score"] for r in group
            ]),
            "fuzzy_states": dict(sorted(Counter(
                r["fuzzy_state"] or "(missing)" for r in group
            ).items())),
            "v2_states": dict(sorted(Counter(
                r["v2_state"] or "(missing)" for r in group
            ).items())),
            "v2_dominant_defects": dict(sorted(Counter(
                r["v2_dominant_defect"] or "(missing)" for r in group
            ).items())),
            "severity_distributions": {
                field: distribution([r[field] for r in group])
                for field in SEVERITY_FIELDS
            },
        }

    score_disagreement = sum(
        r["fuzzy_score"] is not None and
        r["v2_fuzzy_score"] is not None and
        r["fuzzy_score"] != r["v2_fuzzy_score"]
        for r in analysis_rows
    )

    summary = {
        "status": "EXPLORATORY_DAY20_SCORE_SEVERITY_ANALYSIS",
        "input_file": str(INPUT.relative_to(ROOT)),
        "input_record_count": len(analysis_rows),
        "successful_records": len(successful),
        "failed_or_non_success_records": len(analysis_rows) - len(successful),
        "overall_fuzzy_score_distribution": score_distribution,
        "score_state_counts": {
            "fuzzy_state": dict(sorted(Counter(
                r["fuzzy_state"] or "(missing)" for r in analysis_rows
            ).items())),
            "v2_state": dict(sorted(Counter(
                r["v2_state"] or "(missing)" for r in analysis_rows
            ).items())),
        },
        "score_agreement": {
            "fuzzy_score_vs_v2_fuzzy_score_disagreement_count": score_disagreement,
        },
        "by_injected_defect": score_by_defect,
        "correlation_method": (
            "Pearson correlation and Spearman rank correlation with average "
            "ranks for ties. Correlations are calculated only on finite "
            "paired values."
        ),
        "correlations": correlation_rows,
        "interpretation_limits": [
            "All Stage 1 fixtures originate from the same recorded CT source.",
            "All images were synthetic degraded fixtures; no clean controls were identified.",
            "The Stage 1 baseline reported REJECT for all 100 images.",
            "Correlation is descriptive and cannot establish causation.",
            "The existing fuzzy score uses raw image metrics; V2 severities are derived "
            "outputs and are not direct inputs to the current fuzzy score calculation.",
            "Do not change production rules or thresholds based on this report alone.",
            "The locked Day 18 candidate evaluation was not read or rerun.",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)

    with (OUTPUT / "per_image_score_severity.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as f:
        fields = list(analysis_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analysis_rows)

    with (OUTPUT / "correlations.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as f:
        fields = [
            "group", "score_field", "severity_field", "n_pairs",
            "pearson_r", "spearman_rho", "interpretation_note",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(correlation_rows)

    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")

    print("\n=== DAY 20 SCORE-VERSUS-SEVERITY ANALYSIS ===")
    print("Status:", summary["status"])
    print("Baseline rows:", len(analysis_rows))
    print("Successful baseline rows:", len(successful))
    print("Fuzzy score distribution:", json.dumps(score_distribution))
    print("Score disagreement count:", score_disagreement)

    print("\nCorrelations: overall")
    for item in correlation_rows:
        if item["group"] == "overall":
            print(
                f'{item["severity_field"]}: n={item["n_pairs"]}, '
                f'Pearson={item["pearson_r"]}, '
                f'Spearman={item["spearman_rho"]}'
            )

    print("\nScore distribution by defect:")
    for defect, detail in score_by_defect.items():
        print(
            f'{defect}: {detail["score_distribution"]}; '
            f'fuzzy states={detail["fuzzy_states"]}; '
            f'V2 labels={detail["v2_dominant_defects"]}'
        )

    print("\nFiles created:")
    for name in (
        "per_image_score_severity.csv",
        "correlations.csv",
        "summary.json",
    ):
        print(OUTPUT / name)

    print("\nExploratory analysis only. Production code was not modified.")


if __name__ == "__main__":
    main()
