"""
Benchmark with FlashPrefill-matched sparsity levels, varying concurrency (batch).
FlashPrefill sparsity targets (from comparison table):
  4K: 70.4%, 8K: 46.0%, 16K: 29.0%, 32K: 17.6%, 64K: 10.0%
Tests concurrency = 4, 8, 16, 32.
bf16 and fp8, hdim=128 and hdim=256.
MHA (gqa_ratio=1), causal=True, last_n=0.
"""
import torch
import numpy as np
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_block_sparse import (
    get_tile_sizes, build_compact_index_triton_style,
    create_test_inputs, run_fa3, device,
)

torch.manual_seed(42)

# FlashPrefill target sparsity (fraction of K-tiles selected)
FLASH_PREFILL_TARGETS = {
    4096: 70.4,
    8192: 46.0,
    16384: 29.0,
    32768: 17.6,
    65536: 10.0,
}

CONCURRENCY = [4, 8, 16, 32]


def benchmark_fn(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = sorted(times)
    return times[len(times) // 2]


def compute_sparsity(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                     kBlockM, kBlockN, sink, window, last_n, random_blocks):
    """Build index and compute actual sparsity without running attention."""
    bs_cu, bs_idx, _, _ = build_compact_index_triton_style(
        batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=sink, window=window, last_n_blocks=last_n,
        num_random_blocks=random_blocks, causal=True, device=device)
    n_q_tiles = (seqlen_q + kBlockM - 1) // kBlockM
    n_k_tiles = (seqlen_k + kBlockN - 1) // kBlockN
    if seqlen_q == seqlen_k:
        causal_tiles = n_q_tiles * (n_q_tiles + 1) // 2
    else:
        causal_tiles = sum(min(i + 1, n_k_tiles) for i in range(n_q_tiles))
    total_possible = causal_tiles * batch * nheads_kv
    total_selected = bs_idx.shape[0]
    return total_selected / total_possible * 100 if total_possible > 0 else 0


def find_best_config(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                     kBlockM, kBlockN, target_sparsity):
    """Search for (sink, window, last_n=0, random) closest to target sparsity."""
    best = None
    best_diff = 999.0
    for sink in [1, 2, 4]:
        for window in [1, 2, 4, 8, 16]:
            for rand in range(0, 33):
                sp = compute_sparsity(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                                      kBlockM, kBlockN, sink, window, 0, rand)
                diff = abs(sp - target_sparsity)
                if diff < best_diff:
                    best_diff = diff
                    best = (sink, window, 0, rand, sp)
                if diff < 0.5:
                    return best
    return best


def run_bench(dtype, head_dim, batch, seqlen_q, seqlen_k,
              nheads, sink, window, last_n, random_blocks, warmup=3, iters=15):
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)
    softmax_scale = head_dim ** (-0.5)

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch, seqlen_q, seqlen_k, nheads, nheads, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)

    # Dense baseline (no split, since last_n=0)
    fn_dense = lambda: run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                                seqlen_q, head_dim, softmax_scale, causal=True, num_splits=1)
    med_dense = benchmark_fn(fn_dense, warmup, iters)

    # Sparse
    bs_cu, bs_idx, total_q_tiles, cu_q_tiles = build_compact_index_triton_style(
        batch, nheads, nheads, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=sink, window=window, last_n_blocks=last_n,
        num_random_blocks=random_blocks, causal=True, device=device)

    fn_sparse = lambda: run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                                 seqlen_q, head_dim, softmax_scale, causal=True,
                                 block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                                 total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles,
                                 num_splits=1)
    med_sparse = benchmark_fn(fn_sparse, warmup, iters)

    speedup = med_dense / med_sparse if med_sparse > 0 else 0
    return med_dense * 1000, med_sparse * 1000, speedup


def main():
    NHEADS = 8  # MHA
    SEQ_LENS = [4096, 8192, 16384, 32768, 65536]

    for head_dim in [128, 256]:
        for dtype, dtype_name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            element_size = 1 if dtype == torch.float8_e4m3fn else 2
            kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

            print(f"\n{'='*130}")
            print(f"  FlashPrefill Sparsity Benchmark: {dtype_name}, hdim={head_dim}, MHA, kBlockM={kBlockM}, kBlockN={kBlockN}")
            print(f"{'='*130}")

            # Find best matching config for each seq len (use batch=1 for search, sparsity ratio is batch-independent)
            configs = {}
            for sq in SEQ_LENS:
                target = FLASH_PREFILL_TARGETS[sq]
                best = find_best_config(1, NHEADS, NHEADS, sq, sq, kBlockM, kBlockN, target)
                if best is None:
                    print(f"  WARNING: no config found for {sq} target={target}%")
                    continue
                sink, window, ln, rand, actual_sp = best
                configs[sq] = (sink, window, ln, rand, actual_sp)
                print(f"  SeqLen={sq:>6}: target={target:>5.1f}% -> config (sink={sink},win={window},ln=0,rand={rand}) actual={actual_sp:>5.1f}%")

            for sq in SEQ_LENS:
                if sq not in configs:
                    continue
                sink, window, ln, rand, actual_sp = configs[sq]
                target = FLASH_PREFILL_TARGETS[sq]

                print(f"\n  --- SeqLen={sq}, target_sparsity={target:.1f}%, actual={actual_sp:.1f}% (sink={sink},win={window},rand={rand}) ---")
                print(f"  {'Conc':>6} | {'Dense ms':>10} {'Sparse ms':>11} {'Speedup':>9} {'Sparse%':>8}")
                print(f"  {'-'*60}")

                for conc in CONCURRENCY:
                    try:
                        dense_ms, sparse_ms, speedup = run_bench(
                            dtype, head_dim, conc, sq, sq,
                            NHEADS, sink, window, 0, rand, warmup=3, iters=15)
                        print(f"  {conc:>6} | {dense_ms:>9.3f}m {sparse_ms:>10.3f}m {speedup:>8.2f}x {actual_sp:>7.1f}%")
                    except Exception as e:
                        print(f"  {conc:>6} | ERROR: {str(e)[:80]}")
                    torch.cuda.empty_cache()
                print(f"  {'-'*60}")

    print(f"\n{'='*130}")
    print("Notes:")
    print("  - Sparsity matched to FlashPrefill comparison table")
    print("  - MHA (gqa_ratio=1), causal=True, last_n=0, num_splits=1 (no split)")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
