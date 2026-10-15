# Clinical Gatekeeper — Decision Record

## Decision D7 — State/Dominant-Defect Contract Semantics

**Status: PROPOSED / PENDING — Not approved for production changes**

**Date proposed:** 2026-10-15
**Required sign-off:** M1 (agreed in principle), M2 (pending), M3 (pending)

### D7(a) — State and dominant defect are separate quantities

Allow all 15 combinations of three states x five labels.
States: ACCEPT, BORDERLINE, REJECT
Labels: motion_blur, low_contrast, low_snr, high_contrast, none

none is valid only when all directional defect severities are zero.
Current v2_contract.py CONTRADICTS this: must not change until D7(a) confirmed.

M1: agreed in principle. M2: PENDING. M3: PENDING.

### D7(b) — Tie priority and label vocabulary

Canonical label vocabulary: motion_blur, low_contrast, low_snr, high_contrast, none
Note: current code uses sensor_noise/high_noise — NO change until D7(b) signed.
Tie priority: motion_blur > low_contrast > low_snr > high_contrast
high_contrast disabled in production until explicitly approved.

M1: agreed in principle. M2: PENDING. M3: PENDING.

### D7(c) — Neutral, state-aware wording

Wording must not claim metrics prove physical cause.
Domain review by Vishnu Sir required. NOT YET RECORDED.

### Open Items Before D7 Can Be Confirmed

- [ ] M2 sign-off on D7(a) all-zero + any state allowed
- [ ] M2 sign-off on D7(b) low_snr canonical label
- [ ] M3 sign-off on D7(a) and D7(b)
- [ ] Domain reviewer sign-off on D7(c)
- [ ] Dataset cell minimums agreed in writing
- [ ] Rounding rule agreed in writing
