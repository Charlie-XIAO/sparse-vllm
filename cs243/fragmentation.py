#!/usr/bin/env python3
"""Experiment of measuring internal fragmentations."""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

CURRENT_DIR = Path(__file__).parent
ROOT_DIR = CURRENT_DIR.parent
BENCH_DIR = ROOT_DIR / "benchmarks"
LOGS_DIR = CURRENT_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)


def main():
    now = datetime.now()
    now_repr = now.strftime("%Y%m%d-%H%M%S-%f")

    model_name = "facebook/opt-125m"
    dataset_name = "ShareGPT_V3_unfiltered_cleaned_split.json"

    # Make sure that the target dataset exists
    dataset_path = BENCH_DIR / dataset_name
    if not dataset_path.exists():
        print(f"\033[31;1mERROR:\033[0m Dataset not found at: {dataset_path}")
        return

    # Open server log files
    stdout_path = LOGS_DIR / f"fragmentation-{now_repr}.stdout.log"
    stderr_path = LOGS_DIR / f"fragmentation-{now_repr}.stderr.log"
    fout = stdout_path.open("w", encoding="utf-8")
    ferr = stderr_path.open("w", encoding="utf-8")

    # Start the server in the backend and redirect stdout and stderr to files
    print("\033[90mStarting up server...\033[0m")
    server_proc = subprocess.Popen(
        ["vllm", "serve", model_name, "--enforce-eager"],
        cwd=ROOT_DIR,
        env={
            **os.environ, "VLLM_LOGGING_LEVEL": "ERROR",
            "VLLM_CS243_PRINT_FRAGMENTATION": "1"
        },
        stdout=fout,
        stderr=ferr,
    )

    # Wait until the server is up
    max_retries, retry_interval = 12, 5
    time.sleep(10)
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                print(f"\033[32;1m[Att#{i + 1}] Server up\033[0m")
                break
            else:
                print(f"\033[90m[Att#{i + 1}] Server not ready "
                      "({response.status_code})\033[0m")
        except requests.exceptions.ConnectionError:
            print(f"\033[90m[Att#{i + 1}] Server not ready (connection error)"
                  "\033[0m")

        time.sleep(retry_interval)

    # Run the client
    print()
    result = subprocess.run(
        [
            "python", "benchmarks/benchmark_serving.py", "--backend", "vllm",
            "--model", model_name, "--dataset-name", "sharegpt",
            "--dataset-path", dataset_path, "--request-rate", "10",
            "--num-prompts", "1000"
        ],
        cwd=ROOT_DIR,
        env=os.environ,
    )
    print()
    assert result.returncode == 0

    # Shut down the server and close the file descriptors
    print("\033[90mShutting down server...\033[0m")
    time.sleep(3)  # Cool down
    server_proc.terminate()
    fout.close()
    ferr.close()

    # Read server output and create plots
    print("\033[90mProcessing server output...\033[0m")
    result = subprocess.run(
        [
            "python", "cs243/fragmentation_plot.py", "--log-file",
            stdout_path.name
        ],
        cwd=ROOT_DIR,
        env=os.environ,
    )
    assert result.returncode == 0
    print(f"\033[32;1mTimestamp of logs and results: {now_repr}\033[0m")


if __name__ == "__main__":
    main()
