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
    args_list = [args.batch_size, args.dataset]
    if args.dataset == "sharegpt":
        args_list.extend([args.sharegpt_path])
    elif args.dataset == "random":
        args_list.extend([args.random_input_len, args.random_output_len,
                          args.random_range_ratio, args.random_prefix_len])
    args_list.extend([args.sparse_kv_cache_method, args.sparse_kv_cache_budget,
                      args.sparse_kv_cache_num_per_evict,
                      args.sparse_kv_cache_internal])
    args_repr = "bench--" + "-".join(str(arg) for arg in args_list)

    # Make sure that the target dataset exists
    sharegpt_path = None
    if args.dataset == "sharegpt":
        sharegpt_path = BENCH_DIR / args.sharegpt_path
        if not sharegpt_path.exists():
            print("\033[31;1mERROR:\033[0m ShareGPT dataset not found at: "
                  f"{sharegpt_path}")
            return

    # Open server log files
    stdout_path = LOGS_DIR / f"{args_repr}.stdout.log"
    stderr_path = LOGS_DIR / f"{args_repr}.stderr.log"
    metrics_path = LOGS_DIR / f"{args_repr}.metrics.json"
    if (not args.force and stderr_path.exists() and stderr_path.exists()
        and metrics_path.exists()):
        print("\033[33;1mSKIPPED:\033[0m Benchmark outputs already found at:\n"
              f"- {stdout_path}\n"
              f"- {stderr_path}\n"
              f"- {metrics_path}")
        return
    fout = stdout_path.open("w", encoding="utf-8")
    ferr = stderr_path.open("w", encoding="utf-8")

    # Determine server options
    server_options = [
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-seqs",
        str(args.batch_size),
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
        ["vllm", "serve", args.model, *server_options],
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

    # Benchmark configurations
    client_options = [
        "--save-result",
        "--result-filename",
        str(metrics_path),
        "--backend",
        "vllm",
        "--model",
        args.model,
        "--dataset-name",
        args.dataset,
        "--dataset-path",
        str(sharegpt_path),
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.random_output_len),
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--random-prefix-len",
        str(args.random_prefix_len),
        "--request-rate",
        "inf",
        "--num-prompts",
        "1000",
    ]

    # Run the client
    print()
    result = subprocess.run(
        ["python", "benchmarks/benchmark_serving.py", *client_options],
        cwd=ROOT_DIR,
        env=os.environ,
    )
    print()
    assert result.returncode == 0

    # Shut down the server and close the file descriptors
    print("\033[90mShutting down server...\033[0m")
    server_proc.terminate()
    time.sleep(10)  # Cool down
    fout.close()
    ferr.close()

    print(f"Stdout: \033[90m{stdout_path}\033[0m")
    print(f"Stderr: \033[90m{stderr_path}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")

    # Experiment setup arguments
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset", type=str, choices=["sharegpt", "random"],
                        default="sharegpt")

    # ShareGPT dataset options
    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument("--sharegpt-path", type=str,
                                default="sharegpt.json",
                                help="Dataset path relative to /benchmarks.")

    # Random dataset options
    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument("--random-input-len", type=int, default=1024)
    random_group.add_argument("--random-output-len", type=int, default=128)
    random_group.add_argument("--random-range-ratio", type=float, default=1.0)
    random_group.add_argument("--random-prefix-len", type=int, default=0)

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
                        default=512)
    parser.add_argument("--sparse-kv-cache-num-per-evict", type=int, default=1)
    parser.add_argument("--sparse-kv-cache-internal",
                        type=str,
                        choices=["no-op", "free-block", "copy", "spvllm"],
                        default="spvllm")

    args = parser.parse_args()
    if args.sparse_kv_cache_budget == "max":
        assert args.sparse_kv_cache_internal == "no-op"
    main(args)
