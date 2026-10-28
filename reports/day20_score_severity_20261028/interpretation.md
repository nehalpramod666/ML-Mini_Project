# Day 20 — Score Versus Severity Interpretation

**Status:** Exploratory analysis
**Dataset:** Stage 1 baseline, 100 synthetic fixtures
**Production rules and thresholds:** Unchanged

## 1. Overall findings

The fuzzy score distribution was:

- Count: 100
- Minimum: 17
- Median: 24
- Mean: 22.83
- Maximum: 41

All 100 fixtures received the legacy fuzzy state REJECT and the V2 state REJECT. The score in the V2 payload agreed with the fuzzy engine score for every record.

## 2. Pooled correlations

| V2 severity | Pearson r | Spearman rho |
|---|---:|---:|
| Blur | -0.7227 | -0.7917 |
| Contrast | -0.0990 | 0.0082 |
| Noise | 0.2382 | 0.2781 |
| Maximum severity | -0.4899 | -0.0713 |
| Sum of severities | -0.5879 | -0.6310 |

These pooled results must not be interpreted without considering defect-class differences.

## 3. Within-defect findings

### Blur

All 34 blur fixtures had a fuzzy score of 17. Correlations with the severity values are undefined because the score has zero variance within this class.

Nineteen of 34 fixtures were labelled motion_blur and 15 were labelled low_snr in the Stage 1 baseline.

### Contrast

For the 33 contrast fixtures:

- Score versus blur severity: Pearson -0.9572; Spearman -1.0000.
- Score versus contrast severity: Pearson -0.9971; Spearman -1.0000.
- Score versus noise severity: Pearson 0.9156; Spearman 1.0000.

These strong associations are descriptive and apply only to the tested synthetic settings. They do not establish generalization or causal relationships. The simultaneous movement of multiple severities indicates cross-sensitivity.

### Noise

For the 33 noise fixtures:

- Score versus noise severity: Pearson -0.9741; Spearman -1.0000.
- Score versus maximum severity: Pearson -0.9741; Spearman -1.0000.
- Score versus sum of severities: Pearson -0.9741; Spearman -1.0000.

Within this class, higher noise severity was associated with lower score. This differs from the positive pooled correlation, demonstrating the influence of mixing defect classes.

## 4. Architecture recommendation

Provisionally retain the existing fuzzy quality score and V2 defect severities as complementary outputs.

The fuzzy quality score remains the existing M2 engine's output based on raw metrics. V2 severity values remain derived measures that describe particular defect dimensions.

Do not make severities contribute directly to the production score based on this analysis alone.

## 5. Limitations

- All 100 fixtures originate from one recorded CT source.
- Fixtures are synthetic degraded images.
- The manifest contains no explicitly designated clean controls.
- All fixtures received REJECT, limiting assessment of state discrimination.
- Correlation is descriptive and does not establish causation or clinical validity.
- Cross-sensitivity may influence the relationships between raw metrics and severities.

## 6. Next steps

1. Preserve this analysis and the original Stage 1 baseline.
2. Review per-setting score and severity behavior.
3. Investigate cross-sensitivity without changing production rules.
4. Use independent-source and appropriately labelled validation before proposing score redesign.
5. Keep the shared contract unchanged until M1 and M3 provide their feedback.
