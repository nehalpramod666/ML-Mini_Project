# Day 14 — M2 Calibration and Integration Checkpoint

## Status

Engineering investigation and candidate evaluation completed.
Production calibration remains provisional and is not clinically validated.

## Integration verification

The M2 assessment, v2 contract builder, and M1 input adapter were exercised
together on the CT reference and the available RGB ultrasound example.

Both handoffs succeeded and produced strict-JSON-compatible M1 inputs.
This establishes structural compatibility for those examples, not assessment
accuracy or clinical suitability.

## Existing regression tests

- ML tests: 23 passed.
- CV fuzzy tests: 54 passed.
- Nine NumPy/scikit-fuzzy deprecation warnings were non-fatal.

## Candidate evaluation

The candidate evaluator processed 101 images: one original CT reference
and 100 synthetic degraded fixtures. All 100 degraded fixtures originated
from a single source image.

The existing v2 defect labels were:
- Blur: 11 motion_blur, 23 sensor_noise.
- Contrast: 25 low_contrast, 8 sensor_noise.
- Noise: 33 sensor_noise.
- Clean reference: sensor_noise.

All 101 images retained state REJECT under the existing fuzzy score/state
logic.

### Experimental impulse estimator

Settings: median kernel size 3; residual threshold 64; candidate threshold
0.1% of pixels.

On the 100 corrupted fixtures:
- True positives: 33.
- False negatives: 0.
- False positives: 0.
- True negatives: 67.

The minimum estimated impulse percentage among noise fixtures was
0.8301%. The maximum among the clean, blur, and contrast fixtures was 0%.

This is separation on the current synthetic fixture set only. The estimator
targets salt-and-pepper impulse corruption and must not be interpreted as
a general sensor-noise estimator.

### Candidate defect selection

The candidate evaluator compared severity floors 0, 1, 5, and 10.

All four settings matched the expected synthetic labels for all 100
corrupted fixtures. Floor 0 labelled the clean reference motion_blur.
Floors 1, 5, and 10 labelled the clean reference none.

These comparisons are exploratory, use the same source image as the
fixtures, and do not establish independent generalization. No candidate
floor is approved for production.

## Remaining limitations

1. All synthetic degraded fixtures use one source CT image.
2. There is only one clean reference image.
3. The global mean/std SNR proxy does not isolate sensor noise from image
   structure and contrast.
4. The original fuzzy quality score is passed through by v2 rather than
   recalculated from v2 severity values.
5. The current state thresholds and candidate severity formulas have not
   been justified by independent calibration data.
6. A valid contract and successful M1 handoff do not prove the defect
   assessment is correct.
7. Clinical validity has not been established.

## Decision

Close this checkpoint as an engineering investigation and reproducible
candidate evaluation, not as final or clinical calibration approval.

Keep production metrics, fuzzy rules, state thresholds, and v2 conversion
formulas unchanged until candidate changes are justified by suitable
independent evaluation. Retain the generated JSON and CSV reports for review.

Generated reports:
- reports/m2_day14_candidate_evaluation.json
- reports/m2_day14_candidate_per_image.csv
