"""Plotting script for logs generated by fragmentation.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

CURRENT_DIR = Path(__file__).parent
LOGS_DIR = CURRENT_DIR / "logs"
RESULTS_DIR = CURRENT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def main(args):
    assert args.log_file.endswith(".stdout.log")

    num_active_arr, num_total_arr = [], []
    with (LOGS_DIR / args.log_file).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#CS243#,"):
                num_active, num_total = line[8:].split(",")
                num_active_arr.append(int(num_active))
                num_total_arr.append(int(num_total))
    num_active_arr = np.asarray(num_active_arr)
    num_total_arr = np.asarray(num_total_arr)

    frag_prop_arr = 1 - num_active_arr / num_total_arr
    plt.plot(frag_prop_arr[100:])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.grid()
    plt.savefig(RESULTS_DIR / f"{args.log_file[:-11]}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    args = parser.parse_args()
    main(args)
