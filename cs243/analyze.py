#!/usr/bin/env python3
"""Analyze logs generated by the experiments and organize results."""

import json
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).parent
LOGS_DIR = CURRENT_DIR / "logs"
RESULTS_DIR = CURRENT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def analyze(f, name):
    num_active_arr = []
    num_total_arr = []
    copy_overhead_num = 0
    copy_overhead_total = 0

    for line in f:
        if line.startswith("#CS243F#,"):
            num_active, num_total = line[9:].split(",")
            num_active_arr.append(int(num_active))
            num_total_arr.append(int(num_total))
        elif line.startswith("#CS243O#,"):
            copy_overhead_num += 1
            copy_overhead_total += int(line[9:])

    # Strip off warmup and profiling stages
    num_active_arr = np.asarray(num_active_arr[200:])
    num_total_arr = np.asarray(num_total_arr[200:])
    frag_prop_arr = 1 - num_active_arr / num_total_arr

    with (RESULTS_DIR / f"{name}-frag.npy").open("wb") as ffrag:
        np.save(ffrag, num_active_arr)
        np.save(ffrag, num_total_arr)

    return dict(
        frag_prop_max=frag_prop_arr.max(),
        frag_prop_mean=frag_prop_arr.mean(),
        frag_prop_median=np.median(frag_prop_arr),
        frag_prop_p99=np.percentile(frag_prop_arr, 99),
        copy_overhead_num=copy_overhead_num,
        copy_overhead_total=copy_overhead_total,
    )


def append_metrics(f, results):
    metrics = json.load(f)
    for key in (
            "duration",
            "completed",
            "total_input_tokens",
            "total_output_tokens",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "std_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "std_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "std_itl_ms",
            "p99_itl_ms",
    ):
        results[f"__{key}__"] = metrics[key]


def main():
    all_results = {}

    for path in LOGS_DIR.glob("bench--*.stdout.log"):
        name = path.name[7:-11]
        metrics_path = LOGS_DIR / f"bench--{name}.metrics.json"
        if not metrics_path.exists():
            continue

        with path.open("r", encoding="utf-8") as f:
            results = analyze(f, name)
        with metrics_path.open("r", encoding="utf-8") as f:
            append_metrics(f, results)

        all_results[name] = results

    with (RESULTS_DIR / "analyze.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()