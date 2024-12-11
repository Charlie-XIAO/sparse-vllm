#!/usr/bin/env python3
"""Make plots of analyzed metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter

sns.set_theme()

CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "results"
PLOTS_DIR = CURRENT_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

batch_sizes = [2048]
internals = ["no-op", "free-block", "sparse-copy", "spvllm"]
common_key = "sharegpt-sharegpt.json-h2o"


def get_params(key):
    items = key.split("-")
    if items[1] == "sharegpt":
        return dict(
            batch_size=items[0],
            dataset="sharegpt",
            sparse_kv_cache_method=items[3],
            sparse_kv_cache_budget="max"
            if items[4] == "max" else int(items[4]),
            sparse_kv_cache_num_per_evict=int(items[5]),
            sparse_kv_cache_internal="-".join(items[6:]),
        )
    raise NotImplementedError


def plot_metric(data,
                metric,
                metric_repr,
                batch_size,
                formatter="{:.1f}",
                percentage=False):
    mapping = {
        budget: [
            data[f"{batch_size}-{common_key}-{budget}-1-{internal}"][metric]
            for internal in internals
        ]
        for budget in [256, 512, 1024]
    }
    keys = list(mapping.keys())
    values = np.asarray(list(mapping.values()))

    bar_width = 0.2
    x = np.arange(len(keys))
    offset = (len(internals) - 1) * bar_width / 2

    plt.figure()
    for i, internal in enumerate(internals):
        bar_positions = x + i * bar_width - offset
        plt.bar(bar_positions,
                values[:, i],
                bar_width,
                alpha=0.6,
                edgecolor="black",
                label=internal)
        for j, value in enumerate(values[:, i]):
            plt.text(bar_positions[j],
                     value * 1.005,
                     formatter.format(value),
                     ha="center",
                     va="bottom",
                     fontsize=6)
    plt.axhline(data[f"{batch_size}-{common_key}-max-1-no-op"][metric],
                color="black",
                linestyle="--",
                label="vllm")
    plt.xlabel("Budget (tokens)")
    plt.ylabel(metric_repr)
    if percentage:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xticks(x, keys)
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, 1.1),
               ncols=len(internals) + 1,
               fontsize="small",
               columnspacing=0.8)
    plt.tight_layout()
    figname = f"x-metric-{batch_size}-{metric}.png"
    plt.savefig(PLOTS_DIR / figname)
    plt.close()
    print(f"Plotted: {figname}")


def main():
    print("\033[34;1m**********\033[0m")
    with (RESULTS_DIR / "analyze.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    for batch_size in batch_sizes:
        plot_metric(data, "__duration__", "Time taken (s)", batch_size)
        plot_metric(data,
                    "__total_input_tokens__",
                    "Total input tokens",
                    batch_size,
                    formatter="{:.1e}")
        plot_metric(data,
                    "__total_output_tokens__",
                    "Total output tokens",
                    batch_size,
                    formatter="{:.1e}")
        plot_metric(data, "__p99_ttft_ms__", "P99 TTFT (ms)", batch_size)

        for prefix in ["", "eff"]:
            plot_metric(data, f"{prefix}__request_throughput__",
                        "Request throughput (req/s)", batch_size)
            plot_metric(data, f"{prefix}__total_token_throughput__",
                        "Total token throughput (token/s)", batch_size)
            plot_metric(data, f"{prefix}__p99_tpot_ms__", "P99 TPOT (ms)",
                        batch_size)
            plot_metric(data, f"{prefix}__p99_itl_ms__", "P99 ITL (ms)",
                        batch_size)

        plot_metric(data, "num_batched_tokens_mean", "Mean batch size",
                    batch_size)
        plot_metric(data,
                    "num_preempted_total",
                    "Total preempted tokens",
                    batch_size,
                    formatter="{:d}")
        plot_metric(data,
                    "attn_op_total",
                    "Attention forward time (ms)",
                    batch_size,
                    formatter="{:.1e}")
        plot_metric(data,
                    "gpu_utilization_p90",
                    "P90 GPU utilization (%)",
                    batch_size,
                    formatter="{:.1%}",
                    percentage=True)
        plot_metric(data,
                    "gpu_utilization_mean",
                    "Mean GPU utilization (%)",
                    batch_size,
                    formatter="{:.1%}",
                    percentage=True)
        plot_metric(data,
                    "frag_prop_p99",
                    "P99 Internal fragmentation (%)",
                    batch_size,
                    formatter="{:.1%}",
                    percentage=True)
        plot_metric(data,
                    "copy_overhead_total",
                    "Copy overhead (ms)",
                    batch_size,
                    formatter="{:.1e}")


if __name__ == "__main__":
    main()
