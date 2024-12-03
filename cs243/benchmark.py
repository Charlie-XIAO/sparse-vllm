#!/usr/bin/env python3
"""Run benchmark experiments and generate logs."""

import argparse
import os
import subprocess
import time
from pathlib import Path

import requests

CURRENT_DIR = Path(__file__).parent
ROOT_DIR = CURRENT_DIR.parent
BENCH_DIR = ROOT_DIR / "benchmarks"
LOGS_DIR = CURRENT_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)


def main(args):
    print(args)
    args_repr = ("bench--"
                 f"{args.sparse_kv_cache_method}-"
                 f"{args.sparse_kv_cache_budget}-"
                 f"{args.sparse_kv_cache_num_per_evict}-"
                 f"{args.sparse_kv_cache_internal}")

    # Make sure that the target dataset exists
    dataset_path = BENCH_DIR / args.dataset_path
    if not dataset_path.exists():
        print(f"\033[31;1mERROR:\033[0m Dataset not found at: {dataset_path}")
        return

    # Open server log files
    stdout_path = LOGS_DIR / f"{args_repr}.stdout.log"
    stderr_path = LOGS_DIR / f"{args_repr}.stderr.log"
    fout = stdout_path.open("w", encoding="utf-8")
    ferr = stderr_path.open("w", encoding="utf-8")

    # Determine server options
    server_options = [
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-batched-tokens",
        "2048",
        "--max-num-seqs",
        "2048",
        "--sparse-kv-cache-method",
        args.sparse_kv_cache_method,
        "--sparse-kv-cache-budget",
        str(args.sparse_kv_cache_budget),
        "--sparse-kv-cache-num-per-evict",
        str(args.sparse_kv_cache_num_per_evict),
        "--sparse-kv-cache-internal",
        args.sparse_kv_cache_internal,
    ]

    # Start the server in the backend and redirect stdout and stderr to files
    print()
    print("\033[90mStarting up server...\033[0m")
    server_proc = subprocess.Popen(
        ["vllm", "serve", args.model, "--enforce-eager", *server_options],
        cwd=ROOT_DIR,
        env={
            **os.environ, "VLLM_LOGGING_LEVEL": "ERROR",
            "VLLM_CS243_PRINT_BENCHMARK": "1"
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
                      f"({response.status_code})\033[0m")
        except requests.exceptions.ConnectionError:
            print(f"\033[90m[Att#{i + 1}] Server not ready (connection error)"
                  "\033[0m")

        time.sleep(retry_interval)

    # Run the client
    print()
    result = subprocess.run(
        [
            "python", "benchmarks/benchmark_serving.py", "--backend", "vllm",
            "--model", args.model, "--dataset-name", "sharegpt",
            "--dataset-path", dataset_path, "--request-rate", "inf",
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

    print(f"Stdout: \033[90m{stdout_path}\033[0m")
    print(f"Stderr: \033[90m{stderr_path}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment setup arguments
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="The dataset path relative to the benchmarks directory.")

    # KV cache sparsification arguments
    # NOTE(Charlie-XIAO): Set --sparse-kv-cache-budget to max for experimenting
    # with no KV cache eviction. This is because --sparse-kv-cache-method=None
    # will not log the fragmentation information (so it is not allowed here).
    parser.add_argument("--sparse-kv-cache-method",
                        type=str,
                        choices=["random", "h2o"],
                        default="h2o")
    parser.add_argument("--sparse-kv-cache-budget",
                        type=lambda val: int(val) if val != "max" else "max",
                        default="max")
    parser.add_argument("--sparse-kv-cache-num-per-evict", type=int, default=1)
    parser.add_argument("--sparse-kv-cache-internal",
                        type=str,
                        choices=["no-op", "free-block", "copy", "spvllm"],
                        default="spvllm")

    args = parser.parse_args()
    main(args)
