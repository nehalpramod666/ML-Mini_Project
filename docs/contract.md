# Inter-Module Communication Contract (Legacy v1 Specification)

> **Important Contract Evolution Note (Phase 1):**
> This document specifies the **legacy v1** contract shape.
> For the approved Phase 1 contract with strictly typed integer severities (`severity_0_100`),
> canonical v2 labels (`low_snr` instead of legacy `high_noise`), and separate `metrics_raw` tracing,
> see [docs/contract_v2_draft.md](contract_v2_draft.md).
> Downstream components must not confuse the legacy v1 float metrics with v2 calibrated integer severities.

This document describes the original baseline JSON shape passed between
CV+Fuzzy Engine → Backend → Frontend (and → LLM prompt builder).



\## CV + Fuzzy Engine Output (M2 → Backend)



```json

{

&#x20; "metrics": {

&#x20;   "blur": 85.2,

&#x20;   "contrast": 40.1,

&#x20;   "noise": 12.4

&#x20; },

&#x20; "fuzzy\_score": 32,

&#x20; "state": "REJECT",

&#x20; "dominant\_defect": "motion\_blur"

}

```



\- `metrics.blur`: float, variance of Laplacian

\- `metrics.contrast`: float, RMS deviation

\- `metrics.noise`: float, SNR estimate

\- `fuzzy\_score`: int, 0–100

\- `state`: string, one of `"ACCEPT"`, `"BORDERLINE"`, `"REJECT"`

\- `dominant\_defect`: string, e.g. `"motion\_blur"`, `"low\_contrast"`, `"high\_noise"`, `"none"`



\## LLM Engine Input (Backend → M1's LLM)



```json

{

&#x20; "blur": 85,

&#x20; "contrast": 40,

&#x20; "state": "REJECT",

&#x20; "dominant\_defect": "motion\_blur"

}

```



\## LLM Engine Output (M1 → Backend, streamed as plain text tokens)



Plain text, streamed token-by-token. Example:



```

Image rejected due to severe motion artifact. Advise patient to

suspend respiration for the next capture attempt.

```



\## Notes

\- Any change to this contract must be discussed by all 3 members before merging.

\- Last updated: Week 1, Day 5 (per PRD schedule)
