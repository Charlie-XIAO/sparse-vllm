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


def main():
    with (RESULTS_DIR / "analyze.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    for batch_size in [256, 2048]:
        baseline_key = f"{batch_size}-sharegpt-sharegpt.json-h2o-max-1-no-op"
        baseline_throughput = data[baseline_key]["__total_token_throughput__"]

        max_throughput = 0
        for key, value in data.items():
            if key.startswith(f"{batch_size}-"):
                max_throughput = max(max_throughput,
                                     value["__total_token_throughput__"])

        plt.bar(["Raw", "Sparsified"], [baseline_throughput, max_throughput],
                edgecolor="black", alpha=0.6)
        plt.text(0, baseline_throughput + max_throughput * 0.01, "1x",
                 ha="center", va="bottom", fontweight="bold", fontsize=18)
        plt.text(1, max_throughput + max_throughput * 0.01,
                 f"{max_throughput / baseline_throughput:.2f}x", ha="center",
                 va="bottom", fontweight="bold", fontsize=18)
        plt.ylabel("Total token throughput (token/s)")
        plt.ylim(0, max_throughput * 1.2)
        plt.tight_layout()
        figname = f"x-motivation-{batch_size}.png"
        plt.savefig(PLOTS_DIR / figname)
        plt.close()
        print(f"Plotted: {figname}")


if __name__ == "__main__":
    main()