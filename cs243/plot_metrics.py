#!/usr/bin/env python3
"""Make plots of analyzed metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "results"
PLOTS_DIR = CURRENT_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

internals = ["no-op", "free-block", "copy"]
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


def plot_metric(data, metric, metric_repr, batch_size):
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
        plt.bar(x + i * bar_width - offset,
                values[:, i],
                bar_width,
                alpha=0.6,
                edgecolor="black",
                label=internal)
    plt.axhline(data[f"{batch_size}-{common_key}-max-1-no-op"][metric],
                color="black",
                linestyle="--",
                label="vLLM")
    plt.xlabel("Budget (tokens)")
    plt.ylabel(metric_repr)
    plt.xticks(x, keys)
    plt.legend(loc="upper right")
    plt.tight_layout()
    figname = f"x-metric-{batch_size}-{metric}.png"
    plt.savefig(PLOTS_DIR / figname)
    print(f"Plotted: {figname}")


def main():
    with (RESULTS_DIR / "analyze.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    for batch_size in [256, 2048]:
        plot_metric(data, "__request_throughput__", "Total KV tokens",
                    batch_size)
        plot_metric(data, "__total_token_throughput__", "Output tokens",
                    batch_size)
        plot_metric(data, "__p99_ttft_ms__", "P99 TTFT (ms)", batch_size)
        plot_metric(data, "__p99_tpot_ms__", "P99 TPOT (ms)", batch_size)
        plot_metric(data, "__p99_itl_ms__", "P99 ITL (ms)", batch_size)
        plot_metric(data, "frag_prop_p99", "P99 Internal fragmentation (%)",
                    batch_size)


if __name__ == "__main__":
    main()
