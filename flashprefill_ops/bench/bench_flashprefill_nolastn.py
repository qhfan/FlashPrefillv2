"""
Benchmark with FlashPrefill-matched sparsity levels, last_n=0, varying num_splits.
FlashPrefill sparsity targets (from comparison table):
  4K: 70.4%, 8K: 46.0%, 16K: 29.0%, 32K: 17.6%, 64K: 10.0%
last_n=0 (no always-visible K-blocks), no load imbalance.
Tests num_splits = 1, 2, 4, 8.
bf16 and fp8, hdim=128 and hdim=256.
MHA (gqa_ratio=1), causal=True, batch=1.
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

FLASH_PREFILL_TARGETS = {
    4096: 70.4,
    8192: 46.0,
    16384: 29.0,
    32768: 17.6,
    65536: 10.0,
}

SEQ_LENS = [4096, 8192, 16384, 32768, 65536]
SPLIT_VALS = [1, 2, 4, 8]
LAST_N = 0


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
                     kBlockM, kBlockN, target_sparsity, last_n=LAST_N):
    best = None
    best_diff = 999.0
    for sink in [1, 2, 4]:
        for window in [1, 2, 4, 8, 16]:
            for rand in range(0, 33):
                sp = compute_sparsity(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                                      kBlockM, kBlockN, sink, window, last_n, rand)
                diff = abs(sp - target_sparsity)
                if diff < best_diff:
                    best_diff = diff
                    best = (sink, window, last_n, rand, sp)
                if diff < 0.5:
                    return best
    return best


def run_bench(dtype, head_dim, batch, seqlen_q, seqlen_k,
              nheads, sink, window, last_n, random_blocks,
              num_splits=1, warmup=3, iters=15):
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)
    softmax_scale = head_dim ** (-0.5)
    dtype_name = "fp8" if dtype == torch.float8_e4m3fn else "bf16"

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch, seqlen_q, seqlen_k, nheads, nheads, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)

    fn_dense = lambda: run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                                seqlen_q, head_dim, softmax_scale, causal=True, num_splits=1)
    med_dense = benchmark_fn(fn_dense, warmup, iters)

    bs_cu, bs_idx, total_q_tiles, cu_q_tiles = build_compact_index_triton_style(
        batch, nheads, nheads, seqlen_q, seqlen_k,
        kBlockM, kBlockN,
        attention_sink=sink, window=window, last_n_blocks=last_n,
        num_random_blocks=random_blocks, causal=True, device=device)

    fn_sparse = lambda: run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                                 seqlen_q, head_dim, softmax_scale, causal=True,
                                 block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                                 total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles,
                                 num_splits=num_splits)
    med_sparse = benchmark_fn(fn_sparse, warmup, iters)

    n_q_tiles = (seqlen_q + kBlockM - 1) // kBlockM
    n_k_tiles = (seqlen_k + kBlockN - 1) // kBlockN
    total_selected = bs_idx.shape[0]
    if seqlen_q == seqlen_k:
        causal_tiles = n_q_tiles * (n_q_tiles + 1) // 2
    else:
        causal_tiles = sum(min(i + 1, n_k_tiles) for i in range(n_q_tiles))
    total_possible = causal_tiles * batch * nheads
    sparsity_pct = total_selected / total_possible * 100 if total_possible > 0 else 0
    flops_ratio = total_possible / total_selected if total_selected > 0 else 1.0
    speedup = med_dense / med_sparse if med_sparse > 0 else 0

    return {
        "dtype": dtype_name, "hdim": head_dim, "batch": batch,
        "sq": seqlen_q, "num_splits": num_splits,
        "dense_ms": med_dense * 1000, "sparse_ms": med_sparse * 1000,
        "sparsity_pct": sparsity_pct, "flops_ratio": flops_ratio,
        "speedup": speedup,
    }


def main():
    NHEADS = 8
    BATCH = 1

    for head_dim in [128, 256]:
        for dtype, dtype_name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            element_size = 1 if dtype == torch.float8_e4m3fn else 2
            kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

            print(f"\n{'='*160}")
            print(f"  FlashPrefill Sparsity + last_n={LAST_N} Benchmark: {dtype_name}, hdim={head_dim}, MHA, batch={BATCH}")
            print(f"  kBlockM={kBlockM}, kBlockN={kBlockN}, Forced num_splits: {SPLIT_VALS}")
            print(f"{'='*160}")

            configs = {}
            for sq in SEQ_LENS:
                target = FLASH_PREFILL_TARGETS[sq]
                best = find_best_config(1, NHEADS, NHEADS, sq, sq, kBlockM, kBlockN, target)
                if best is None:
                    print(f"  WARNING: no config found for {sq} target={target}%")
                    continue
                sink, window, ln, rand, actual_sp = best
                configs[sq] = (sink, window, ln, rand, actual_sp)
                print(f"  SeqLen={sq:>6}: target={target:>5.1f}% -> config (sink={sink},win={window},ln={ln},rand={rand}) actual={actual_sp:>5.1f}%")

            for sq in SEQ_LENS:
                if sq not in configs:
                    continue
                sink, window, ln, rand, actual_sp = configs[sq]
                target = FLASH_PREFILL_TARGETS[sq]

                print(f"\n  --- SeqLen={sq}, target={target:.1f}%, actual={actual_sp:.1f}% (sink={sink},win={window},ln={ln},rand={rand}) ---")

                hdr = f"  {'Splits':>7} | {'Sparse%':>7} {'FLOPs_R':>7} | {'Dense ms':>9} {'Sparse ms':>10} {'Speedup':>8}"
                print(hdr)
                print(f"  {'-'*70}")

                for ns in SPLIT_VALS:
                    try:
                        r = run_bench(dtype, head_dim, BATCH, sq, sq,
                                      NHEADS, sink, window, ln, rand,
                                      num_splits=ns, warmup=3, iters=15)
                        print(f"  sp={ns:>5} | {r['sparsity_pct']:>6.1f}% {r['flops_ratio']:>6.1f}x | {r['dense_ms']:>8.3f}m {r['sparse_ms']:>9.3f}m {r['speedup']:>7.2f}x")
                    except Exception as e:
                        print(f"  sp={ns:>5} | ERROR: {str(e)[:80]}")
                    torch.cuda.empty_cache()
                print(f"  {'-'*70}")

    print(f"\n{'='*160}")
    print("Notes:")
    print(f"  - Sparsity matched to FlashPrefill comparison table, last_n={LAST_N} (no always-visible K-blocks)")
    print("  - MHA (gqa_ratio=1), causal=True, batch=1")
    print("  - Dense baseline always uses num_splits=1")
    print("  - FLOPs_R: theoretical FLOPs ratio (causal), Sparse%: % K-tiles selected")
    print(f"{'='*160}")

if __name__ == "__main__":
    main()
