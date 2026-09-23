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

## Training format (Unsloth / Llama 3.2 conversational format)

{
  "messages": [
    {
      "role": "system",
      "content": "<system-style task description>"
    },
    {
      "role": "user",
      "content": "<metrics JSON as string>"
    },
    {
      "role": "assistant",
      "content": "<clinical instruction>"
    }
  ]
}

> Note: `contrast` is treated as bidirectional in the synthetic dataset (lower and higher values represent different defect directions, with an optimal mid-range); `blur` and `noise` are unidirectional, where higher values indicate greater degradation. This synthetic convention must be confirmed against M2's real metric/fuzzy implementation.