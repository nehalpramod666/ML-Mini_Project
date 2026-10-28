# Contract V2 Proposal — M2 / M1 / M3

**Status:** DRAFT — awaiting M1 and M3 review
**Prepared:** 27 October 2026
**Target:** Agree on the interface required for M3 `fuzzy_infer` by Day 22
**Authority:** This proposal is not the approved shared contract. Do not replace `docs/contract.md` until the owners sign off.

## 1. Purpose and current evidence

This proposal clarifies the difference between raw image measurements, derived defect severities, the overall fuzzy quality score, and the canonical defect label.

The Day 19 exploratory Stage 1 baseline processed 100 synthetic fixtures from one recorded CT source image:

- Blur: 19/34 correctly isolated (55.8824%).
- Contrast reduction: 25/33 correctly isolated (75.7576%).
- Noise: 33/33 correctly isolated (100%).
- Overall: 77/100 correctly isolated (77%).
- All 100 legacy fuzzy states and V2 states were REJECT.
- No processing errors occurred in that run.

These are exploratory single-source synthetic-fixture results, not clinical validation. The baseline does not authorize changes to production rules or thresholds.

## 2. Proposed distinction: raw metrics versus severities

### 2.1 Raw measurements

The `metrics_raw` object carries measurements computed from the input image. These values are not normalized to a 0–100 severity scale.

| Proposed field | Meaning | Interpretation |
|---|---|---|
| `metrics_raw.blur` | Variance of the Laplacian | Higher values often indicate sharper edges, but high-frequency noise can also raise this value. |
| `metrics_raw.contrast` | Standard deviation of image intensities | Measures intensity spread; it is not, by itself, a universal measure of image quality. |
| `metrics_raw.snr_proxy` | Existing mean-intensity / standard-deviation ratio | An SNR-like proxy; higher generally means a better signal-to-variability ratio. It is not a specific sensor-noise estimate or a dB measurement. |

**Proposal:** Rename the raw field `metrics_raw.noise` to `metrics_raw.snr_proxy` in an approved schema revision. If owners decide that compatibility requires retaining `metrics_raw.noise`, its meaning must explicitly state that it is the raw SNR-like proxy, not a noise severity.

### 2.2 Derived severities

The existing `metrics` object contains derived 0–100 severity values. To minimize disruption to consumers, this proposal provisionally retains that object name but explicitly defines its meaning.

| Field | Meaning | Direction |
|---|---|---|
| `metrics.blur` | Blur severity | Higher = worse blur |
| `metrics.contrast` | Published contrast severity | Higher = greater contrast-related defect severity |
| `metrics.noise` | Noise-related severity derived from the SNR-like proxy | Higher = worse noise-related quality |

All three severity values should be finite numbers on the 0–100 scale. A raw metric must never be substituted for its severity in the same field.

`metrics_scale` must unambiguously identify the 0–100 severity scale, using the approved value from the implementation.

The existing `metrics` name is retained here provisionally for compatibility. M1 and M3 must confirm whether this is sufficiently clear or whether a future schema should use an explicitly named `severities` object.

## 3. Proposed label mapping

The canonical M2/V2 noise-related defect label is **`low_snr`**.

| Origin or consumer | Label | Proposed handling |
|---|---|---|
| Canonical V2 | `low_snr` | Canonical representation in the V2 payload |
| M1 legacy model/input adapter | `sensor_noise` | Explicit adapter mapping from V2 `low_snr` |
| Legacy M2 fuzzy engine | `high_noise` | Legacy label; translate explicitly at a documented adapter boundary, not as a second canonical V2 label |
| No dominant defect | `none` | Valid only when the V2 contract's no-defect conditions are satisfied |

M1 must confirm that its adapter maps V2 severities to the model's expected input fields and maps `low_snr` to `sensor_noise` where required.

Unknown labels must not be silently converted to `none`.

## 4. Disabled high-contrast trigger

The current V2 payload builder raises `ContractV2Error` with code `DISABLED_HIGH_CONTRAST_TRIGGER` when high-contrast severity would win or tie the highest enabled dominant-defect candidate while the production trigger remains disabled.

**Proposal: preserve this as an explicit error. Do not return `dominant_defect: none`.**

Returning `none` would risk representing a detected but unsupported condition as a successful no-defect outcome.

The current production trigger remains disabled. Enabling it requires a separate decision, approved scope, and versioned evaluation.

Under the current implementation, `none` is permitted only when all directional severities are zero and the state is `ACCEPT`. Inconsistent combinations should produce an explicit consistency error.

M1 and M3 must agree that consumers will treat the disabled-trigger condition as an error, not as a valid defect-free payload.

## 5. Authority for `fuzzy_score` and `state`

M2's fuzzy engine remains the proposed authority for the overall quality score and quality state.

| Field | Meaning |
|---|---|
| `fuzzy_score` | Integer quality score from 0 to 100; higher means better overall quality |
| `state` | Quality state derived from `fuzzy_score` using the agreed thresholds |

The current thresholds remain unchanged:

- `ACCEPT`: score >= 75
- `BORDERLINE`: score >= 45 and < 75
- `REJECT`: score < 45

The V2 builder may validate that `state` agrees with `fuzzy_score`; it must not independently recalibrate or silently replace the score produced by the fuzzy engine.

Defect severity has the opposite direction from the quality score: higher severity means worse defect-related quality, while a higher fuzzy score means better overall quality.

Any future change to score semantics or thresholds requires explicit owner approval and a separate versioned evaluation. The Day 19 baseline did not authorize such changes.

## 6. Proposed non-finite value handling

The raw metrics and JSON payload must follow explicit rules.

| Input condition | Proposed contract behavior |
|---|---|
| Finite, nonnegative blur or contrast value | Serialize as a JSON number |
| Finite, nonnegative SNR-like proxy | Serialize as a JSON number; status `finite` |
| Positive-infinite SNR established from the source image | Serialize raw proxy as JSON `null`, status `positive_infinity`, and noise severity 0 |
| Undefined SNR, including an undefined 0/0 condition | Return a structured error; do not treat it as infinite SNR |
| NaN, negative infinity, negative raw measurements, or inconsistent metric/image values | Reject with an explicit error rather than serialize an invalid numeric value |

The current V2 builder already distinguishes positive-infinite SNR from an undefined source-image ratio and validates metric/image consistency. M1 and M3 should review these rules against the exact edge cases covered by `cv_fuzzy/v2_contract.py` tests before sign-off.

The fuzzy engine's existing handling of some non-finite SNR inputs must not be confused with the public contract's validation rules.

JSON output must remain standards-compliant: do not emit literal `NaN` or `Infinity` tokens.

## 7. Proposed `fuzzy_infer` interface for M3

### 7.1 Function signature

Proposed Python interface:

`fuzzy_infer(image: np.ndarray) -> dict[str, Any]`

Input:

- A nonempty, two-dimensional grayscale NumPy array representing the decoded image.
- The caller is responsible for passing a decoded image; DICOM loading remains the responsibility of the existing image-loading path.
- Unsupported dimensions or empty input must produce an explicit input error.

Processing:

1. Run the existing `analyze_image_fuzzy(image)` function.
2. Pass that result and the same source image to `build_v2_payload(m2_result, image)`.
3. Return the valid V2 success payload without silently changing the score, state, severities, or dominant-defect label.

### 7.2 Success result

The success payload must contain the approved version of these fields:

- `contract_version`
- `metrics_scale`
- `metrics_raw`
- `metrics_raw_status`
- `metrics`
- `fuzzy_score`
- `state`
- `dominant_defect`

`metrics_raw` contains raw measurements; `metrics` contains 0–100 severity values.

### 7.3 Error behavior

**Proposal:** Errors must remain explicit and must never be converted to a successful `dominant_defect: none` result.

- Preserve `FuzzyInferenceError` for fuzzy inference failures.
- Preserve `ContractV2Error` and its error code for V2 validation/payload failures.
- M3 must catch these errors at its integration boundary and present an appropriate unavailable/error state instead of claiming the image has no defect.
- Unexpected exceptions should be logged and surfaced as processing failures; internal stack traces should not be presented as normal user-facing output.

The exact wrapping strategy and whether M3 wants exceptions or an explicit result envelope must be confirmed with M3 before implementation. This proposal does not claim that a new `fuzzy_infer` wrapper already exists.

### 7.4 Acceptance tests for the interface

Before sign-off, verify:

1. A valid grayscale image produces all required success fields.
2. Raw measurements remain separate from severity values.
3. Score and state agree with the 75/45 thresholds.
4. `low_snr` is translated to `sensor_noise` only at the agreed M1 adapter boundary.
5. Disabled high contrast produces an explicit error, not a false `none`.
6. Positive-infinite and undefined SNR cases follow the agreed rules.
7. Invalid image shapes and inference failures are handled explicitly.
8. V2 payload creation does not silently alter production fuzzy rules or thresholds.

## 8. Compatibility and versioning

This is a proposal, not a live schema change.

The current code uses `metrics_raw.noise` for the raw SNR-like value and `metrics.blur`, `metrics.contrast`, and `metrics.noise` for severity values.

Renaming `metrics_raw.noise` to `metrics_raw.snr_proxy` changes the field expected by consumers. The owners must approve either:

1. A versioned rename with an explicit migration; or
2. Retention of the existing field name with a mandatory, unambiguous definition.

Do not change `docs/contract.md`, the production payload builder, or the M1 adapter until agreement is documented.

## 9. Required M1 and M3 decisions

Please explicitly review and sign off on:

1. Raw field naming and the distinction between raw measurements and severities.
2. Noise direction and the proposed raw name `snr_proxy`.
3. Canonical V2 label `low_snr`, the M1 alias `sensor_noise`, and legacy `high_noise` handling.
4. The disabled high-contrast error behavior.
5. M2 ownership of `fuzzy_score` and `state`, including the existing 75/45 thresholds.
6. Non-finite value and error behavior.
7. The `fuzzy_infer(image: np.ndarray) -> dict[str, Any]` interface, success payload and error-handling convention.
8. The compatibility/versioning plan.

**Target:** Obtain M1/M3 sign-off by Day 22 so the agreed `fuzzy_infer` interface can be implemented and integrated in time.

After approval, update `docs/contract.md` in a separately reviewed change. Until then, this document remains a proposal and the production rules and thresholds remain frozen.
