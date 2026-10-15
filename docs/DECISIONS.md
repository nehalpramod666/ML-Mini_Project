# Clinical Gatekeeper — Decision Record

## Decision D7 — State/Dominant-Defect Contract Semantics (Phase 1)

**Status: APPROVED (Phase 1 Contract Alignment)**

**Date approved:** 2026-10-15
**Sign-offs:**
- **M2 (Computer Vision / Fuzzy Engine Lead):** APPROVED.
  - *Rationale:* Decouples fuzzy quality score from dominant defect categorization, establishes deterministic half-even integer rounding on `[0, 100]` directional severities, standardizes external canonical label `low_snr`, safely gates `high_contrast` behind a production error trigger, and ensures structural integration without making unverified clinical claims.
- **M3 (FastAPI Backend / Integration Lead):** APPROVED.
  - *Rationale:* Provides an unambiguous, strictly typed success contract (`contract_version: 2`) with integer severities and no ambiguous error fields in success payloads. Standardizes error codes (`UNDEFINED_SNR`, `DISABLED_HIGH_CONTRAST_TRIGGER`, `CONSISTENCY_ERROR`) for clean mapping to HTTP response envelopes at the API layer.
- **Domain Review (Clinical / Terminology):** PENDING.
  - *Status:* Interim neutral measurement wording approved in principle for engineering documentation and LLM prompt context; formal user-facing clinical wording remains pending domain sign-off.

> **Note on Calibration:** Approvals represent architectural and structural integration agreement. Provisional calibration formulas remain engineering candidates and are not clinically validated.

---

### D7(a) — State and Dominant Defect Separation

1. `state` is derived exclusively from the integer `fuzzy_score`:
   - `ACCEPT`: score $\ge 75$
   - `BORDERLINE`: $45 \le \text{score} < 75$
   - `REJECT`: score $< 45$
2. `dominant_defect` is selected from directional defect-severity candidates.
3. All-zero directional severities require `state="ACCEPT"` and `dominant_defect="none"`.
4. All-zero severities with `BORDERLINE` or `REJECT` produce a structured consistency error (`CONSISTENCY_ERROR`), not a successful v2 payload.
5. Non-ACCEPT states with nonzero defect severity are permitted, subject to dominant-label consistency.
6. Assessment errors remain strictly separate from the successful v2 JSON payload. No `error` or `status` field is added to the strict success schema.

---

### D7(b) — Canonical Vocabulary and LLM Boundary Adapter

1. **Canonical v2 labels:**
   - `motion_blur`
   - `low_contrast`
   - `high_contrast` (trigger disabled in production)
   - `low_snr` (canonical external label)
   - `none`
2. **LLM Boundary Adapter:**
   - The model's training vocabulary contains `sensor_noise`.
   - The LLM engine (`ml/llm_engine.py`) explicitly maps incoming canonical `low_snr` $\to$ `sensor_noise` at the inference boundary.
   - Legacy label `high_noise` is strictly rejected; never silently mapped.
   - `sensor_noise` is rejected as a canonical label in nested v2 payloads (it is an internal training vocabulary token, not a canonical v2 contract label).
   - Training dataset vocabulary is preserved without modification or regeneration.
3. **Neutral Description:**
   - Report generation and contextual instruction describe this category neutrally as an SNR-related measurement or proxy, explicitly avoiding claims that physical sensor hardware malfunction has been confirmed.
   - Base training system prompt (`SYSTEM_PROMPT`) is preserved identically to maintain training-prompt parity.

---

### Severity Representation, Rounding, and Dominance

1. `metrics_raw` retains raw floating-point measurements with original precision.
2. `metrics_raw_status` preserves strict value/status relationships (`finite` vs `positive_infinity`).
3. `metrics` contains exactly `blur`, `contrast`, and `noise`. Every published severity must be an integer from 0 to 100 inclusive. Fractional values and booleans are rejected.
4. Scale interpretation: higher severity (0 to 100) indicates worse measured quality. `fuzzy_score` (0 to 100) indicates better overall quality.
5. **Rounding rule:** Each derived directional severity candidate is clamped to `[0.0, 100.0]`, then rounded once using Python's half-even convention (`int(round(clamped))`). Consumers must not re-round published severity integers.
6. **Tie priority for enabled labels:**
   $$\text{motion\_blur} > \text{low\_contrast} > \text{low\_snr}$$
7. **Contrast handling:**
   - Directional candidates are low-contrast severity and high-contrast severity.
   - `high_contrast` trigger remains disabled in production.
   - If the rounded high-contrast candidate is positive and ties or exceeds the largest enabled candidate, raise `ContractV2Error` with code `DISABLED_HIGH_CONTRAST_TRIGGER`.
   - If positive but strictly lower than the largest enabled candidate, the enabled winner is selected.
   - Published `metrics.contrast` is $\max(\text{low\_contrast\_s}, \text{high\_contrast\_s})$. No `high_contrast` field is added to the schema.

---

### Structured Error Codes

`ContractV2Error` defines stable, machine-readable error codes:
- `UNDEFINED_SNR`: 0/0 intensity mean and standard deviation.
- `DISABLED_HIGH_CONTRAST_TRIGGER`: High contrast candidate tied or exceeded enabled candidates while trigger is disabled.
- `CONSISTENCY_ERROR`: All-zero severity with non-ACCEPT state, state/score mismatch, or label/severity mismatch.
- `INVALID_SCHEMA`: Type, key, or value-range violation.

---

### Dataset Reporting Floor

- Minimum exploratory held-out reporting floor: $\ge 20$ independent source-image groups per state/label cell for any reported performance claim.
- Multiple degraded variants of one original image count as one source group.
- Underrepresented cells must be marked insufficient rather than pooled.
- Training coverage, held-out evaluation coverage, and API fixture coverage are tracked as separate requirements.
- Independent-source acquisition remains an outstanding work item because existing evaluation fixtures derive from a single CT source.
