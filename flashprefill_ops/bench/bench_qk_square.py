"""
Benchmark block sparse attention speedup vs dense, with Q==K (square attention).
Tests: 1K, 2K, 4K, 8K, 16K, 32K, 64K seq lengths.
Different sparsity levels via varying sink/window/last_n/random.
bf16 and fp8, hdim=128 and hdim=256.
MHA (gqa_ratio=1), batch=1.
Reports speedup for NoSplit and Split(auto) modes.
"""
import torch
import numpy as np
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_block_sparse import (
    get_tile_sizes, build_full_compact_index, build_compact_index_triton_style,
    create_test_inputs, run_fa3, device,
)

torch.manual_seed(42)

def benchmark_fn(fn, warmup=5, iters=20):
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

def run_bench(dtype, head_dim, batch, seqlen_q, seqlen_k,
              nheads, sink, window, last_n, random_blocks,
              num_splits=1, warmup=5, iters=20):
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)
    softmax_scale = head_dim ** (-0.5)
    dtype_name = "fp8" if dtype == torch.float8_e4m3fn else "bf16"

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch, seqlen_q, seqlen_k, nheads, nheads, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)

    # Dense baseline (always num_splits=1 for fair comparison)
    fn_dense = lambda: run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                                seqlen_q, head_dim, softmax_scale, causal=True,
                                num_splits=1)
    med_dense = benchmark_fn(fn_dense, warmup, iters)

    # Sparse with forced num_splits
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

    # Sparsity stats
    n_q_tiles = (seqlen_q + kBlockM - 1) // kBlockM
    n_k_tiles = (seqlen_k + kBlockN - 1) // kBlockN
    total_selected = bs_idx.shape[0]
    # Causal: dense only computes lower-triangle (including diagonal)
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
        "sq": seqlen_q, "sk": seqlen_k, "nheads": nheads,
        "num_splits": num_splits, "sink": sink, "window": window,
        "last_n": last_n, "rand": random_blocks,
        "dense_ms": med_dense * 1000, "sparse_ms": med_sparse * 1000,
        "sparsity_pct": sparsity_pct, "flops_ratio": flops_ratio,
        "speedup": speedup,
    }

def main():
    SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    NHEADS = 8  # MHA
    BATCH = 1
    SPLIT_VALS = [1, 2, 4, 8]  # forced num_splits to test

    # Sparsity configs: (label, sink, window, last_n, random)
    # last_n=0 to disable last_n, test pure block sparse without load imbalance
    configs = [
        ("ultra-sparse", 1, 1, 0, 0),
        ("sparse",       2, 2, 0, 1),
        ("medium",       2, 4, 0, 2),
        ("medium-r4",    2, 4, 0, 4),
    ]

    for head_dim in [128, 256]:
        for dtype, dtype_name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            print(f"\n{'='*160}")
            print(f"  Q==K Benchmark: {dtype_name}, hdim={head_dim}, heads={NHEADS}, batch={BATCH}")
            print(f"  Forced num_splits: {SPLIT_VALS}")
            print(f"{'='*160}")

            for label, sink, window, last_n, rand in configs:
                print(f"\n  --- {label}: sink={sink},win={window},ln={last_n},rand={rand} ---")
                # Header
                hdr = f"  {'SeqLen':>8} | {'Sparse%':>7} {'FLOPs_R':>7} |"
                for ns in SPLIT_VALS:
                    hdr += f" SP={ns:>1}ms  SPup={ns:>1} |"
                print(hdr)
                print(f"  {'-'*140}")

                for sq in SEQ_LENS:
                    sk = sq
                    try:
                        # Run with num_splits=1 first to get sparsity stats
                        r1 = run_bench(dtype, head_dim, BATCH, sq, sk,
                                       NHEADS, sink, window, last_n, rand,
                                       num_splits=1, warmup=3, iters=15)
                        row = f"  {sq:>8} | {r1['sparsity_pct']:>6.1f}% {r1['flops_ratio']:>6.1f}x |"
                        row += f" {r1['sparse_ms']:>7.3f} {r1['speedup']:>7.2f}x |"

                        for ns in SPLIT_VALS[1:]:  # skip 1, already done
                            r = run_bench(dtype, head_dim, BATCH, sq, sk,
                                          NHEADS, sink, window, last_n, rand,
                                          num_splits=ns, warmup=3, iters=15)
                            row += f" {r['sparse_ms']:>7.3f} {r['speedup']:>7.2f}x |"
                        print(row)
                    except Exception as e:
                        print(f"  {sq:>8} | ERROR: {str(e)[:100]}")
                print(f"  {'-'*140}")

    print(f"\n{'='*160}")
    print("Notes:")
    print("  - Q==K (square attention), MHA (gqa_ratio=1), batch=1")
    print("  - Dense baseline always uses num_splits=1")
    print(f"  - SP=Nms: sparse time (ms) with forced num_splits=N, SPup=N: speedup vs dense")
    print("  - FLOPs_R: theoretical FLOPs ratio (causal), Sparse%: % K-tiles selected")
    print(f"{'='*160}")

if __name__ == "__main__":
    main()
