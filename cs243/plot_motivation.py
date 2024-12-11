#!/usr/bin/env python3
"""Make motivation plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "results"
PLOTS_DIR = CURRENT_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

batch_sizes = [2048]


def main():
    print("\033[34;1m**********\033[0m")
    with (RESULTS_DIR / "analyze.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    motivation_key = "eff__request_throughput__"
    for batch_size in batch_sizes:
        baseline_key = f"{batch_size}-sharegpt-sharegpt.json-h2o-max-1-no-op"
        baseline_throughput = data[baseline_key][motivation_key]

        max_throughput = 0
        for key, value in data.items():
            if key.startswith(f"{batch_size}-"):
                max_throughput = max(max_throughput, value[motivation_key])

        plt.bar(["w/o sparsification", "w sparsification"],
                [baseline_throughput, max_throughput],
                edgecolor="black",
                linewidth=3,
                alpha=0.6)
        plt.text(0,
                 baseline_throughput + max_throughput * 0.01,
                 "1x",
                 ha="center",
                 va="bottom",
                 fontweight="bold",
                 fontsize=18)
        plt.text(1,
                 max_throughput + max_throughput * 0.01,
                 f"{max_throughput / baseline_throughput:.2f}x",
                 ha="center",
                 va="bottom",
                 fontweight="bold",
                 fontsize=18)
        plt.annotate("",
                     xy=(1, max_throughput),
                     xytext=(0, baseline_throughput - 0.05 * max_throughput),
                     arrowprops={
                         "arrowstyle": "->",
                         "lw": 5,
                         "alpha": 0.6,
                         "color": "black"
                     })
        plt.ylabel("Request throughput (req/s)")
        plt.ylim(0, max_throughput * 1.2)
        plt.tight_layout()
        figname = f"x-motivation-{batch_size}.png"
        plt.savefig(PLOTS_DIR / figname)
        plt.close()
        print(f"Plotted: {figname}")


if __name__ == "__main__":
    main()
