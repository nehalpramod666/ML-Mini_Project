import statistics
import time

from ml.llm_engine import generate_report

test_cases = [
    {
        "blur": 85.2,
        "contrast": 40.1,
        "noise": 12.4,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    },
    {
        "blur": 20.0,
        "contrast": 85.0,
        "noise": 15.0,
        "state": "REJECT",
        "dominant_defect": "high_contrast",
    },
    {
        "blur": 15.0,
        "contrast": 50.0,
        "noise": 20.0,
        "state": "ACCEPT",
        "dominant_defect": "none",
    },
    {
        "blur": 70.0,
        "contrast": 48.0,
        "noise": 90.0,
        "state": "REJECT",
        "dominant_defect": "sensor_noise",
    },
    {
        "blur": 45.0,
        "contrast": 40.0,
        "noise": 30.0,
        "state": "BORDERLINE",
        "dominant_defect": "low_contrast",
    },
]


def benchmark_batch(label):
    latencies = []
    print(f"\n --- {label} ---")
    for case in test_cases:
        start = time.perf_counter()
        output = generate_report(case)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        print(
            f"{case['dominant_defect']:14s} | "
            f"{case['state']:10s} | "
            f"{elapsed:.3f}s | "
            f"{output}"
        )
    
    print(f"\n{label} summary:")
    print(f"Mean: {statistics.mean(latencies):.3f}s")
    print(f"Min: {min(latencies):.3f}s")
    print(f"Max: {max(latencies):.3f}s")

    return latencies

if __name__ == "__main__":
    print("LLM-only latency benchmark")
    print("Generation settings: max_tokens=60, temperature=0.0")
    print("Context: n_ctx=1024")

    cold_latencies = benchmark_batch("First-pass inference")
    warm_latencies = benchmark_batch("Warm inference")
    print("\n======== Final COMPARISON ==========")
    print((f"First-pass mean: {statistics.mean(cold_latencies):.3f}s"))
    print(f"Warm mean: {statistics.mean(warm_latencies):.3f}s")