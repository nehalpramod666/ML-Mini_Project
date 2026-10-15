# M2 Fuzzy Scoring Diagnostic

## 1. Scope and status

- Baseline commit: f0236f9
- Baseline branch: develop
- Working branch: feature/m2-fuzzy-scoring-diagnostic
- Purpose: explain observed fuzzy scores, examine rule coverage, and document reproducible runtime behaviour.
- Status: exploratory engineering diagnosis; not clinically validated.

Production files and thresholds remain unchanged. The tests below reproduce behaviour in the existing fuzzy inference system.

## 2. Existing dataset evidence

The committed report `reports/v2_calibration_diagnostic_summary.json` identifies its results as `EXPLORATORY_SINGLE_SOURCE_NOT_CLINICALLY_VALIDATED`. Its input fixture report is marked `DRAFT_FIXTURES_NOT_YET_REVIEWED`.

The dataset contains 101 successfully evaluated fixtures derived from one underlying CT source and synthetic degradations.

| Statistic | Original fuzzy score |
|---|---:|
| Count | 101 |
| Minimum | 17 |
| 25th percentile | 17 |
| Median | 18 |
| 75th percentile | 19 |
| Maximum | 41 |
| REJECT classifications | 101/101 |

The same report records identical original and v2 fuzzy scores for all 101 fixtures. V2 changes the dominant-defect selections, but does not improve score separation in this evaluation.

These results do not establish general image-quality or clinical performance.

## 3. Metric definitions

The implementation in `cv_fuzzy/metrics.py` defines:

- `compute_blur`: variance of the Laplacian. Higher values indicate more high-frequency structure; lower values indicate a blurrier image under this measure.
- `compute_contrast`: standard deviation of pixel intensities. This measures intensity spread, not diagnostic adequacy.
- `compute_noise`: mean pixel intensity divided by standard deviation. This is an SNR proxy, not a direct noise estimator. Lower values indicate a lower signal-to-variation ratio, but image content also affects the result.
- A zero standard deviation returns infinite SNR.

The SNR proxy alone cannot distinguish sensor noise from anatomical variation or other intensity changes.
**Clarification of blur membership:** The `blur["low"]` term represents a low raw blur metric value, not a low membership degree. Because the raw metric is the variance of the Laplacian, a low value has high membership in `blur["low"]` and indicates greater blur under this measure. Conversely, higher Laplacian variance generally indicates more high-frequency image structure and a sharper image. The interpretation remains metric-specific and is not a claim of clinical image quality.

## 4. Fuzzy rules and output semantics

The current engine defines these rules:

| Rule | Condition | Output |
|---|---|---|
| 1 | Low blur membership | Poor quality |
| 2 | Low contrast membership | Poor quality |
| 3 | Low SNR membership | Poor quality |
| 4 | Medium blur AND medium contrast AND medium SNR | Fair quality |
| 5 | High blur AND medium contrast AND high SNR | Good quality |
| 6 | High blur AND high contrast AND high SNR | Good quality |
| 7 | Medium blur AND high contrast AND high SNR | Good quality |

The output quality universe is 0–100. The engine defuzzifies the combined output and rounds the result to an integer.

State thresholds are:

- Score >= 75: `ACCEPT`
- Score >= 45 and < 75: `BORDERLINE`
- Score < 45: `REJECT`

There is no general fallback rule for combinations that do not activate any listed rule.

## 5. Runtime reproduction: clean reference

The existing diagnostic report records these raw values for the clean source:

| Input | Value |
|---|---:|
| Blur | 108.161969 |
| Contrast | 46.938109 |
| SNR proxy | 2.035261 |

Using these inputs directly in the existing fuzzy inference system reproduced the following membership degrees:

| Input term | Membership |
|---|---:|
| Low blur | 0.0000 |
| Medium blur | 0.0368 |
| High blur | 0.9387 |
| Low contrast | 0.0000 |
| Medium contrast | 0.4031 |
| High contrast | 0.3469 |
| Low SNR | 0.4647 |
| Medium SNR | 0.2676 |
| High SNR | 0.0000 |

The runtime result was:

- Raw defuzzified score: 22.0130
- Rounded score: 22
- State: `REJECT`

The low-SNR rule activates poor quality at approximately 0.4647. The fair rule has an approximate activation of 0.0368, limited by the medium-blur membership. The good-quality rules have zero activation because high-SNR membership is zero.

This reproduces the reported clean-source result. It identifies the low-SNR rule as an important contributor but does not establish that the source image contains severe noise.

## 6. Runtime reproduction: uncovered rule combination

A second inference test used the following hypothetical inputs:

| Input | Value |
|---|---:|
| Blur | 200 |
| Contrast | 70 |
| SNR proxy | 3.0 |

Observed memberships:

| Input term | Membership |
|---|---:|
| Low blur | 0.0 |
| Medium blur | 0.0 |
| High blur | 1.0 |
| Low contrast | 0.0 |
| Medium contrast | 0.0 |
| High contrast | 1.0 |
| Low SNR | 0.0 |
| Medium SNR | 0.75 |
| High SNR | 0.0 |

No defined rule activates for this combination:

- None of the three poor-quality rules activates because all low memberships are zero.
- The fair rule does not activate because medium blur and medium contrast are zero.
- The good-quality rules do not activate because high SNR is zero.

The runtime test consequently produced `KeyError: 'quality'` when attempting to read the quality output. No quality score was available for this input.

This confirms a rule-combination coverage gap in the current rule base. It does not demonstrate a gap in every individual membership function: the observed memberships cover their respective inputs, but the rule combinations do not cover every possible combination.

## 7. Findings and limitations

### Verified findings

1. The clean-source score of 22 is reproducible with the current engine.
2. The low-SNR membership activates the poor-quality rule for the clean reference.
3. At least one input combination produces no quality output because no rule activates.
4. The existing evaluation assigns `REJECT` to all 101 fixtures.
5. Original and v2 scores are identical for all 101 fixtures in the existing report.

### Limitations

- The evaluation uses one underlying CT source with synthetic degradations.
- The fixture report is marked as a draft that has not yet been reviewed.
- The SNR proxy is not specific to sensor noise.
- The experimental mapping of injected noise to `sensor_noise` is not approved ground truth.
- These findings do not establish clinical validity or justify new production thresholds.
- No candidate scoring formula has been adopted.

## 8. Recommended follow-up

Before any production changes:

1. Preserve this report as the diagnostic baseline.
2. Add regression coverage for the reproduced clean-source behaviour and the uncovered rule combination.
3. Evaluate metrics on independent source images, keeping each source grouped with its degraded variants.
4. Decide the intended semantics of quality score versus defect severity using reviewed evidence.
5. Test any approved rule or scoring changes separately against the unchanged baseline.

No production code, thresholds, shared contracts, or metric definitions were changed during this diagnostic investigation.

## 9. Independent reproduction

The following PowerShell command runs the existing fuzzy engine without modifying production code. Run it from the repository root with the project's Python environment active.

```powershell
@'
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from cv_fuzzy import fuzzy_engine as fe


def memberships(var, value):
    return {
        term: round(
            float(fuzz.interp_membership(
                var.universe, var[term].mf, value
            )),
            4,
        )
        for term in var.terms
    }


def probe(name, blur, contrast, snr):
    print(f"\n=== {name} ===")
    print(f"Inputs: blur={blur}, contrast={contrast}, snr={snr}")
    print("Blur memberships:", memberships(fe._BLUR, blur))
    print("Contrast memberships:", memberships(fe._CONTRAST, contrast))
    print("SNR memberships:", memberships(fe._SNR, snr))

    simulation = ctrl.ControlSystemSimulation(fe._CONTROL_SYSTEM)
    simulation.input["blur"] = blur
    simulation.input["contrast"] = contrast
    simulation.input["snr"] = snr

    try:
        simulation.compute()
        raw_score = float(simulation.output["quality"])
    except Exception as exc:
        print(f"Inference failed: {type(exc).__name__}: {exc}")
        return

    score = int(round(np.clip(raw_score, 0, 100)))
    state = (
        "ACCEPT" if score >= 75
        else "BORDERLINE" if score >= 45
        else "REJECT"
    )
    print(f"Raw defuzzified score: {raw_score:.4f}")
    print(f"Rounded score: {score}")
    print(f"State: {state}")


probe(
    "Clean reference",
    108.16196905076504,
    46.93810981929438,
    2.035261142873869,
)

probe(
    "Uncovered rule combination",
    200.0,
    70.0,
    3.0,
)
'@ | python -
```

Expected findings:

- **Clean reference:** raw score approximately `22.0130`, rounded score `22`, state `REJECT`.
- **Uncovered rule combination:** the inference system cannot produce the `quality` output and reports `KeyError: 'quality'` when the script attempts to retrieve it.

The script uses internal attributes of the existing fuzzy engine for diagnostic purposes; they are not presented as a public API. The input values and expected findings are those already reproduced during this investigation.

These are engineering reproductions, not clinical validation. Regression tests and evaluation against independent source images remain follow-up tasks.
