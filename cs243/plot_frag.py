#!/usr/bin/env python3
"""Make internal fragmentation micro-benchmark plots."""

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

internals = ["no-op", "free-block", "copy"]


def main():
    for batch_size in [256, 2048]:
        filename = f"{batch_size}-sharegpt-sharegpt.json-h2o-max-1-no-op-frag.npy"
        with (RESULTS_DIR / filename).open("rb") as f:
            baseline_num_active_arr = np.load(f)
            baseline_num_total_arr = np.load(f)
        baseline_frag_prop_arr = 1 - baseline_num_active_arr / baseline_num_total_arr

        for budget in [256, 512, 1024]:
            frag_prop_arrs = []
            for internal in internals:
                filename = (f"{batch_size}-sharegpt-sharegpt.json-h2o-{budget}-1-"
                            f"{internal}-frag.npy")
                with (RESULTS_DIR / filename).open("rb") as f:
                    num_active_arr = np.load(f)
                    num_total_arr = np.load(f)
                frag_prop_arr = 1 - num_active_arr / num_total_arr
                frag_prop_arrs.append(frag_prop_arr)

            plt.figure()
            plt.plot(np.arange(len(baseline_frag_prop_arr)), baseline_frag_prop_arr,
                     color="black", linestyle="--", label="vLLM")
            for internal, frag_prop_arr in zip(internals, frag_prop_arrs):
                x = np.arange(len(frag_prop_arr))
                plt.plot(x, frag_prop_arr, label=internal)
                plt.fill_between(x, 0, frag_prop_arr, alpha=0.2)
            plt.xlabel("Step Number")
            plt.ylabel("Internal Fragmentation (%)")
            plt.ylim(0, 0.8)
            plt.title(f"Budget: {budget} tokens")
            plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
            plt.grid()
            plt.legend(loc="upper right")
            plt.grid()
            figname = f"x-frag-{batch_size}-{budget}.png"
            plt.savefig(PLOTS_DIR / figname)
            print(f"Plotted: {figname}")


if __name__ == "__main__":
    main()
