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
    # System information
    num_scheduled_seqs_arr = []
    num_batched_tokens_arr = []
    num_blocks_to_swap_in_arr = []
    num_blocks_to_swap_out_arr = []
    num_blocks_to_copy_arr = []
    num_blocks_to_migrate_arr = []
    num_slots_to_migrate_arr = []
    running_queue_size_arr = []
    num_preempted_arr = []
    gpu_utilization_arr = []

    # Attention information
    attn_op_num = 0
    attn_op_total = 0

    # Fragmentation information
    num_active_arr = []
    num_total_arr = []

    # Sparse copying information
    copy_overhead_num = 0
    copy_overhead_total = 0

    for line in f:
        # System information
        if line.startswith("#CS243S#,"):
            (_, num_scheduled_seqs, _, num_batched_tokens,
             num_blocks_to_swap_in, num_blocks_to_swap_out, num_blocks_to_copy,
             num_blocks_to_migrate, num_slots_to_migrate, _, _, _,
             running_queue_size, num_preempted,
             gpu_utilization) = line[9:].split(",")
            num_scheduled_seqs_arr.append(int(num_scheduled_seqs))
            num_batched_tokens_arr.append(int(num_batched_tokens))
            num_blocks_to_swap_in_arr.append(int(num_blocks_to_swap_in))
            num_blocks_to_swap_out_arr.append(int(num_blocks_to_swap_out))
            num_blocks_to_copy_arr.append(int(num_blocks_to_copy))
            num_blocks_to_migrate_arr.append(int(num_blocks_to_migrate))
            num_slots_to_migrate_arr.append(int(num_slots_to_migrate))
            running_queue_size_arr.append(int(running_queue_size))
            num_preempted_arr.append(int(num_preempted))
            gpu_utilization_arr.append(float(gpu_utilization))

        # Attention information
        elif line.startswith("#CS243A#,"):
            attn_op_num += 1
            attn_op_total += int(line[9:])

        # Fragmentation information
        elif line.startswith("#CS243F#,"):
            num_active, num_total = line[9:].split(",")
            num_active_arr.append(int(num_active))
            num_total_arr.append(int(num_total))

        # Sparse copying information
        elif line.startswith("#CS243O#,"):
            copy_overhead_num += 1
            copy_overhead_total += int(line[9:])

    # System information
    with (RESULTS_DIR / f"{name}-sys.npy").open("wb") as fsys:
        np.save(fsys, num_scheduled_seqs_arr)
        np.save(fsys, num_batched_tokens_arr)
        np.save(fsys, num_blocks_to_swap_in_arr)
        np.save(fsys, num_blocks_to_swap_out_arr)
        np.save(fsys, num_blocks_to_copy_arr)
        np.save(fsys, num_blocks_to_migrate_arr)
        np.save(fsys, num_slots_to_migrate_arr)
        np.save(fsys, running_queue_size_arr)
        np.save(fsys, num_preempted_arr)
        np.save(fsys, gpu_utilization_arr)
    gpu_utilization_arr = gpu_utilization_arr[500:-1000]

    # Fragmentation information
    # Strip off warmup and profiling stages
    num_active_arr = np.asarray(num_active_arr[200:])
    num_total_arr = np.asarray(num_total_arr[200:])
    frag_prop_arr = 1 - num_active_arr / num_total_arr
    with (RESULTS_DIR / f"{name}-frag.npy").open("wb") as ffrag:
        np.save(ffrag, num_active_arr)
        np.save(ffrag, num_total_arr)

    return dict(
        num_batched_tokens_mean=np.mean(num_batched_tokens_arr),
        num_batched_tokens_median=np.median(num_batched_tokens_arr),
        num_batched_tokens_p99=np.percentile(num_batched_tokens_arr, 99),
        num_preempted_total=sum(num_preempted_arr),
        gpu_utilization_mean=np.mean(gpu_utilization_arr),
        gpu_utilization_p90=np.percentile(gpu_utilization_arr, 90),
        gpu_utilization_p99=np.percentile(gpu_utilization_arr, 99),
        attn_op_num=attn_op_num,
        attn_op_total=attn_op_total / 1e6,
        frag_prop_max=frag_prop_arr.max(),
        frag_prop_mean=frag_prop_arr.mean(),
        frag_prop_median=np.median(frag_prop_arr),
        frag_prop_p99=np.percentile(frag_prop_arr, 99),
        copy_overhead_num=copy_overhead_num,
        copy_overhead_total=copy_overhead_total / 1e6,
    )


def append_metrics(f, results):
    metrics = json.load(f)
    factor = results["gpu_utilization_mean"]

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

    for key in (
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
    ):
        results[f"eff__{key}__"] = metrics[key] / factor

    for key in (
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
        results[f"eff__{key}__"] = metrics[key] * factor


def main():
    all_results = {}

    for path in LOGS_DIR.glob("bench--*.stdout.log"):
        print(f"Analyzing {path.name}...")
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
