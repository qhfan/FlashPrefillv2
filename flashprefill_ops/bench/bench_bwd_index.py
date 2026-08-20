"""Runtime breakdown baseline for the bwd reverse index builder."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_block_sparse_bwd import build_packgqa_csr_varlen
from flash_block_sparse_bwd_index import (
    build_block_sparse_bwd_index,
    _entry_metadata,
    _sort_entries_within_segments,
    _bs_bwd_count_kernel,
    _bs_bwd_fill_kernel,
)

device = "cuda"


def bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def run(sq, gqa=4, nheads_kv=8, sink=1, window=2):
    nheads_q = nheads_kv * gqa
    seqlens = [sq]
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
    bs_cu, bs_idx, total_q_tiles, cu_q_tiles, _ = build_packgqa_csr_varlen(
        seqlens, seqlens, nheads_q, nheads_kv, sink, window, device=device)
    E = bs_idx.numel()
    num_bwd_segments = nheads_kv * 1 * ((sq + 63) // 64)

    # Full pipeline
    t_all = bench(lambda: build_block_sparse_bwd_index(
        bs_cu, bs_idx, cu_q_tiles, cu_q, cu_k, nheads_q, nheads_kv, total_q_tiles, sq))

    # Breakdown: metadata
    t_meta = bench(lambda: _entry_metadata(bs_cu, bs_idx, cu_q_tiles, nheads_kv, total_q_tiles))

    # Breakdown: count + cumsum + fill
    def count_fill():
        h_kv, b, local_p = _entry_metadata(bs_cu, bs_idx, cu_q_tiles, nheads_kv, total_q_tiles)
        bwd_cu = torch.zeros(num_bwd_segments + 1, dtype=torch.int32, device=device)
        _bs_bwd_count_kernel[(E,)](h_kv, b, bs_idx, bwd_cu, 1, (sq + 63) // 64, E, EXPAND=2)
        counts = bwd_cu[1:].clone()
        bwd_cu[1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
        bwd_idx = torch.empty(E * 2, dtype=torch.int32, device=device)
        offsets = bwd_cu[:-1].clone()
        _bs_bwd_fill_kernel[(E,)](h_kv, b, local_p, bs_idx, offsets, bwd_idx, 1, (sq + 63) // 64, E, EXPAND=2)
        return bwd_cu, bwd_idx
    t_cf = bench(count_fill)

    # Breakdown: sort (based on the output of one count_fill run)
    bwd_cu, bwd_idx = count_fill()
    t_sort = bench(lambda: _sort_entries_within_segments(bwd_cu, bwd_idx, num_bwd_segments))

    print(f"sq={sq:>6} gqa={gqa} E={E:>7} | total={t_all:>8.1f}us | "
          f"meta={t_meta:>6.1f} count+fill={t_cf:>6.1f} sort={t_sort:>6.1f} (breakdown repeats metadata)")


if __name__ == "__main__":
    for sq in [4096, 16384, 32768, 65536]:
        run(sq)
    run(32768, gqa=12, nheads_kv=2)
