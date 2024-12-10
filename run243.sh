#!/bin/bash

################################################################################

./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 256 --sparse-kv-cache-internal no-op
./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 256 --sparse-kv-cache-internal free-block
./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 256 --sparse-kv-cache-internal copy

./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 512 --sparse-kv-cache-internal no-op
./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 512 --sparse-kv-cache-internal free-block
./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 512 --sparse-kv-cache-internal copy

./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 1024 --sparse-kv-cache-internal no-op
./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 1024 --sparse-kv-cache-internal free-block
./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 1024 --sparse-kv-cache-internal copy

./cs243/benchmark.py --batch-size 256 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget max --sparse-kv-cache-internal no-op

################################################################################

./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 256 --sparse-kv-cache-internal no-op
./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 256 --sparse-kv-cache-internal free-block
./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 256 --sparse-kv-cache-internal copy

./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 512 --sparse-kv-cache-internal no-op
./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 512 --sparse-kv-cache-internal free-block
./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 512 --sparse-kv-cache-internal copy

./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 1024 --sparse-kv-cache-internal no-op
./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 1024 --sparse-kv-cache-internal free-block
./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget 1024 --sparse-kv-cache-internal copy

./cs243/benchmark.py --batch-size 2048 --sparse-kv-cache-num-per-evict 1 --sparse-kv-cache-budget max --sparse-kv-cache-internal no-op

################################################################################

./cs243/analyze.py
./cs243/plot_frag.py
./cs243/plot_metrics.py
