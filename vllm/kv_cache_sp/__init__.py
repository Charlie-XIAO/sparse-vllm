from vllm.kv_cache_sp.h2o import H2OKVCacheSparsifier
from vllm.kv_cache_sp.random import RandomKVCacheSparsifier


def get_kv_cache_sparsifier(method: str):
    if method == "random":
        return RandomKVCacheSparsifier
    if method == "h2o":
        return H2OKVCacheSparsifier
    raise ValueError(
        f"Unrecognized KV cache sparsification method: {method!r}")


__all__ = [
    "get_kv_cache_sparsifier",
]
