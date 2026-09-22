# Dataset Schema

## Input (matches shared contract, Section 1.9)
{
  "blur": float,        # 0-100, variance-of-Laplacian derived score
  "contrast": float,    # 0-100, RMS-deviation derived score
  "noise": float,        # 0-100, SNR-derived score
  "state": "ACCEPT" | "BORDERLINE" | "REJECT",
  "dominant_defect": "motion_blur" | "low_contrast" | "high_contrast" | "sensor_noise" | "none"
}

## Output
A single string: plain-language clinical instruction.
- 1-2 sentences max
- Names the defect
- Gives a concrete corrective action for retake (not generic "please retake")
- No hedging language ("might be", "possibly") — the fuzzy engine already made the decision

## Training format (Unsloth instruction-tuning, decided now to avoid rework later)
{"instruction": "<system-style task description>", "input": "<metrics JSON as string>", "output": "<clinical instruction>"}