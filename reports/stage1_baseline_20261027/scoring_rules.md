# Stage 1 Gold-Standard Isolation Baseline — Scoring Rules

**Study date:** 2026-10-27
**Study status:** Exploratory baseline
**Production rules and thresholds:** Frozen; no changes permitted in this study

## 1. Dataset

The manifest at `data/gold_standard/manifest.json` contains 100 synthetic degraded PNG fixtures:

- `blur`: 34 images
- `contrast`: 33 images
- `noise`: 33 images

All 100 manifest-referenced images were present at the time of inspection.

No records were explicitly identified as clean, original, control, or defect-free. The baseline therefore evaluates injected-defect isolation only and will not claim clean-image specificity or a clinical false-positive rate.

All records identify the same source path, `cv_fuzzy/sample_data/CT_small.dcm`. Results are exploratory and do not establish clinical validity or generalization.

## 2. Frozen mapping

The expected V2 dominant-defect label is determined from the manifest's `injected_defect` field:

| Injected defect | Expected V2 dominant-defect label |
|---|---|
| `blur` | `motion_blur` |
| `contrast` | `low_contrast` |
| `noise` | `low_snr` |

This mapping must not be changed after observing the evaluation results.

The legacy fuzzy-engine dominant-defect label and V2 dominant-defect label will be recorded separately. Only the V2 label is used for the primary isolation measurement.

## 3. Primary criterion

The PRD target is 100% exact isolation of injected defects.

For each injected-defect class:

Isolation rate = number of manifest records with the exact expected V2 label / total manifest records for that injected-defect class.

Overall isolation rate = number of manifest records with the exact expected V2 label / all 100 manifest records.

A V2 payload error, missing prediction, or incorrect dominant-defect label counts as not isolated. Errors must also be enumerated and reported separately.

The denominator includes all applicable manifest records, including records whose V2 payload fails.

The report must show the numerator, denominator, and percentage for each defect class and overall.

## 4. Confusion matrix

The confusion matrix compares the manifest-derived expected class with the actual V2 prediction.

Expected classes are `motion_blur`, `low_contrast`, and `low_snr`.

Actual prediction columns must preserve observed V2 labels, including labels such as `motion_blur`, `low_contrast`, `low_snr`, `high_contrast`, `none`, and any other observed label. Failed payloads must appear in an explicit error category, with individual error codes recorded in the per-image results.

Do not discard errors or recategorize unexpected predictions to improve the result.

## 5. State and score reporting

For every image, record the available fuzzy score and state from `analyze_image_fuzzy`, plus the V2 payload state and score when payload creation succeeds.

Report the state distribution and V2 errors separately. If the fuzzy engine succeeds but V2 payload construction fails, retain the fuzzy engine result rather than erasing it.

The existing score and state thresholds remain unchanged: `ACCEPT` at scores of 75 or more, `BORDERLINE` at scores of 45 through 74, and `REJECT` below 45.

## 6. Error handling

Each image is processed independently. A loading, fuzzy-inference, or V2 payload error must be recorded for that image without stopping the remaining evaluation.

Record the error stage, error type or code where available, and message. Do not suppress, repair, or rerun failed cases as part of this baseline.

## 7. Evaluation restrictions

- Do not edit `cv_fuzzy/metrics.py`, `cv_fuzzy/fuzzy_engine.py`, or `cv_fuzzy/v2_contract.py`.
- Do not change production thresholds or fuzzy rules.
- Do not enable the disabled high-contrast trigger.
- Do not rerun or modify the locked Day 18 candidate evaluation.
- Do not alter this mapping or the pass criterion after observing the results.

The baseline will be labelled **exploratory baseline** regardless of the measured percentage. The observed result will be compared with the PRD's 100% target without implying clinical validation.
