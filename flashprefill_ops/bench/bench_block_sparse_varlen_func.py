"""
Benchmark / correctness check for block-sparse flash_attn_varlen_func.

Mimics bench_block_sparse_packgqa.py and bench_block_sparse_nopage.py, but the
target API is the training-style varlen entry:

    flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)

Sparse index format is PackGQA CSR, same as flash_attn_with_kvcache:
    segment_id = total_q_tiles * h_kv + (cu_q_tiles[b] + packed_m_block)

Checks:
  1. dense varlen_func vs dense flash_attn_with_kvcache(non-paged)
  2. sparse varlen_func vs sparse flash_attn_with_kvcache(non-paged)
  3. dense varlen_func vs sparse varlen_func approximation error
  4. dense/sparse varlen_func speed, with sparse with_kvcache as reference
"""

import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from test_block_sparse import get_tile_sizes, m_block_to_q_pos, q_pos_to_k_block
from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

torch.manual_seed(42)
device = "cuda"

ATTENTION_SINK = 1
WINDOW = 2
LAST_N_BLOCK = 0
RANDOM_BLOCKS = 4

NHEADS_Q = 32
NHEADS_KV = 8
GQA_RATIO = NHEADS_Q // NHEADS_KV

# Same style as bench_block_sparse_nopage.py: (seqlen_q, seqlen_k) per request.
PROFILES = {
    "short":  [(1024, 4096), (2048, 8192), (1536, 6144), (3072, 12288)],
    "medium": [(4096, 16384), (2048, 12288), (6144, 24576), (3072, 16384)],
    "long":   [(8192, 32768), (4096, 28672), (6144, 32768), (3072, 24576)],
    "xlong":  [(8192, 65536), (4096, 49152), (12288, 65536), (6144, 49152)],
}

LOG_FILE = os.path.join(
    script_dir, f"bench_block_sparse_varlen_func_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
_log_lines = []


def log_print(msg=""):
    print(msg)
    _log_lines.append(msg)


def create_inputs(profile, nheads_q, nheads_kv, head_dim, dtype):
    """Create packed varlen_func inputs and equivalent non-paged KV-cache inputs."""
    batch = len(profile)
    seqlens_q = [p[0] for p in profile]
    seqlens_k = [p[1] for p in profile]
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    scale = 0.5

    q = (torch.randn(total_q, nheads_q, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)
    k = (torch.randn(total_k, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)
    v = (torch.randn(total_k, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)

    cu_seqlens_q = torch.tensor([0] + np.cumsum(seqlens_q).tolist(), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + np.cumsum(seqlens_k).tolist(), dtype=torch.int32, device=device)

    # Same K/V data, but laid out as non-paged KV cache for flash_attn_with_kvcache.
    k_cache = torch.zeros(batch, max_seqlen_k, nheads_kv, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros(batch, max_seqlen_k, nheads_kv, head_dim, dtype=dtype, device=device)
    for b in range(batch):
        k0, k1 = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        sk = seqlens_k[b]
        k_cache[b, :sk] = k[k0:k1]
        v_cache[b, :sk] = v[k0:k1]
    cache_seqlens = torch.tensor(seqlens_k, dtype=torch.int32, device=device)

    return (q, k, v, cu_seqlens_q, cu_seqlens_k, k_cache, v_cache, cache_seqlens,
            seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k)


def build_varlen_packgqa_index(
    seqlens_q, seqlens_k, nheads_q, nheads_kv, kBlockM, kBlockN,
    attention_sink=ATTENTION_SINK, window=WINDOW, last_n_blocks=LAST_N_BLOCK,
    num_random_blocks=RANDOM_BLOCKS, causal=True, device="cuda", rng_seed=42,
):
    """PackGQA CSR index: segment = total_q_tiles * h_kv + (cu_q_tiles[b] + m_block)."""
    gqa_ratio = nheads_q // nheads_kv
    batch = len(seqlens_q)

    q_tiles_per_batch = [
        (seqlens_q[b] * gqa_ratio + kBlockM - 1) // kBlockM for b in range(batch)
    ]
    total_q_tiles = sum(q_tiles_per_batch)
    cu_q_tiles = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    for b in range(batch):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + q_tiles_per_batch[b]

    rng = np.random.RandomState(rng_seed)
    positions_per_m_block = max(1, kBlockM // gqa_ratio)

    all_indices = []
    cu_offsets = [0]
    for _h_kv in range(nheads_kv):
        for b in range(batch):
            sq = seqlens_q[b]
            sk = seqlens_k[b]
            prefix_len = sk - sq
            n_q_tiles = q_tiles_per_batch[b]
            n_k_tiles = (sk + kBlockN - 1) // kBlockN
            n_q_pos_blocks = (sq + positions_per_m_block - 1) // positions_per_m_block
            last_n_q_pos_start = max(0, n_q_pos_blocks - last_n_blocks)

            for m_block in range(n_q_tiles):
                q_pos_start = m_block_to_q_pos(m_block, kBlockM, gqa_ratio)
                if q_pos_start >= sq:
                    cu_offsets.append(len(all_indices))
                    continue
                q_k_blk = q_pos_to_k_block(q_pos_start, prefix_len, kBlockN)
                causal_max_n = min(q_k_blk + 1, n_k_tiles) if causal else n_k_tiles
                q_pos_blk_idx = q_pos_start // positions_per_m_block
                is_last_n = q_pos_blk_idx >= last_n_q_pos_start

                selected = set()
                selected.update(range(min(attention_sink, causal_max_n)))
                selected.update(range(max(0, q_k_blk - window + 1), min(q_k_blk + 1, causal_max_n)))
                if is_last_n:
                    selected.update(range(causal_max_n))
                else:
                    remaining = [kk for kk in range(causal_max_n) if kk not in selected]
                    if remaining and num_random_blocks > 0:
                        n_sel = min(num_random_blocks, len(remaining))
                        pick = np.linspace(0, len(remaining) - 1, n_sel, dtype=int)
                        selected.update(remaining[i] for i in pick)

                all_indices.extend(sorted(selected))
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


def run_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
               softmax_scale, causal=True, block_sparse=None, num_splits=1):
    kwargs = dict(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=causal,
        num_splits=num_splits,
    )
    if block_sparse is not None:
        bs_cu, bs_idx, total_q_tiles, cu_q_tiles = block_sparse
        kwargs.update(
            block_sparse_cu=bs_cu,
            block_sparse_idx=bs_idx,
            total_q_tiles=total_q_tiles,
            cu_q_tiles=cu_q_tiles,
        )
    return flash_attn_varlen_func(**kwargs)


def run_kvcache(q, k_cache, v_cache, cache_seqlens, cu_seqlens_q, max_seqlen_q,
                softmax_scale, causal=True, block_sparse=None, num_splits=1):
    kwargs = dict(
        q=q, k_cache=k_cache, v_cache=v_cache,
        page_table=None, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale, causal=causal,
        num_splits=num_splits,
    )
    if block_sparse is not None:
        bs_cu, bs_idx, total_q_tiles, cu_q_tiles = block_sparse
        kwargs.update(
            block_sparse_cu=bs_cu,
            block_sparse_idx=bs_idx,
            total_q_tiles=total_q_tiles,
            cu_q_tiles=cu_q_tiles,
        )
    return flash_attn_with_kvcache(**kwargs)


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


def max_abs_diff(a, b):
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def rel_error(a, b):
    a_f, b_f = a.to(torch.float32), b.to(torch.float32)
    num = (a_f - b_f).norm().item()
    den = b_f.norm().item()
    return num / den if den > 0 else num


def compute_sparsity(seqlens_q, seqlens_k, bs_idx, kBlockM, kBlockN):
    total_selected = bs_idx.shape[0]
    total_causal = 0
    for b in range(len(seqlens_q)):
        sq, sk = seqlens_q[b], seqlens_k[b]
        prefix_len = sk - sq
        n_q_tiles = (sq * GQA_RATIO + kBlockM - 1) // kBlockM
        n_k_tiles = (sk + kBlockN - 1) // kBlockN
        for m_block in range(n_q_tiles):
            q_pos = m_block_to_q_pos(m_block, kBlockM, GQA_RATIO)
            if q_pos >= sq:
                continue
            q_k = q_pos_to_k_block(q_pos, prefix_len, kBlockN)
            total_causal += min(q_k + 1, n_k_tiles)
    total_causal *= NHEADS_KV
    return total_selected / total_causal * 100 if total_causal > 0 else 0.0


def run_one(dtype, head_dim, profile_name, profile, warmup=3, iters=15):
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=False)
    softmax_scale = head_dim ** (-0.5)
    dtype_name = "fp8" if dtype == torch.float8_e4m3fn else ("fp16" if dtype == torch.float16 else "bf16")

    (q, k, v, cu_seqlens_q, cu_seqlens_k, k_cache, v_cache, cache_seqlens,
     seqlens_q, seqlens_k, max_seqlen_q, max_seqlen_k) = create_inputs(
        profile, NHEADS_Q, NHEADS_KV, head_dim, dtype)

    block_sparse = build_varlen_packgqa_index(
        seqlens_q, seqlens_k, NHEADS_Q, NHEADS_KV, kBlockM, kBlockN, device=device)

    # Correctness: varlen_func vs existing with_kvcache(non-paged) path.
    out_dense_var = run_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               softmax_scale, causal=True)
    out_dense_kv = run_kvcache(q, k_cache, v_cache, cache_seqlens, cu_seqlens_q, max_seqlen_q,
                               softmax_scale, causal=True)
    diff_dense = max_abs_diff(out_dense_var, out_dense_kv)

    out_sparse_var = run_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                softmax_scale, causal=True, block_sparse=block_sparse)
    out_sparse_kv = run_kvcache(q, k_cache, v_cache, cache_seqlens, cu_seqlens_q, max_seqlen_q,
                                softmax_scale, causal=True, block_sparse=block_sparse)
    diff_sparse = max_abs_diff(out_sparse_var, out_sparse_kv)
    rel_approx = rel_error(out_dense_var, out_sparse_var)

    # Speed: target API is varlen_func; with_kvcache(non-paged) is the reference column.
    fn_dense_var = lambda: run_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                      softmax_scale, causal=True)
    fn_sparse_var = lambda: run_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale, causal=True, block_sparse=block_sparse)
    fn_sparse_kv = lambda: run_kvcache(q, k_cache, v_cache, cache_seqlens, cu_seqlens_q, max_seqlen_q,
                                       softmax_scale, causal=True, block_sparse=block_sparse)

    med_dense_var = benchmark_fn(fn_dense_var, warmup, iters) * 1000
    med_sparse_var = benchmark_fn(fn_sparse_var, warmup, iters) * 1000
    med_sparse_kv = benchmark_fn(fn_sparse_kv, warmup, iters) * 1000

    return {
        "dtype": dtype_name,
        "hdim": head_dim,
        "profile": profile_name,
        "kBlockM": kBlockM,
        "kBlockN": kBlockN,
        "total_q": sum(seqlens_q),
        "max_sk": max_seqlen_k,
        "sparsity_pct": compute_sparsity(seqlens_q, seqlens_k, block_sparse[1], kBlockM, kBlockN),
        "diff_dense": diff_dense,
        "diff_sparse": diff_sparse,
        "rel_approx": rel_approx,
        "dense_var_ms": med_dense_var,
        "sparse_var_ms": med_sparse_var,
        "sparse_kv_ms": med_sparse_kv,
    }


def main():
    log_print(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log will be saved to: {LOG_FILE}")
    log_print(f"Target API: flash_attn_varlen_func, PackGQA CSR, GQA Q={NHEADS_Q} KV={NHEADS_KV} ratio={GQA_RATIO}")
    log_print(f"Sparse pattern: sink={ATTENTION_SINK}, window={WINDOW}, last_n={LAST_N_BLOCK}, random={RANDOM_BLOCKS}")

    for head_dim in [128, 256]:
        for dtype, dtype_name in [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            element_size = 1 if dtype == torch.float8_e4m3fn else 2
            kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=False)

            log_print(f"\n{'=' * 180}")
            log_print(f"  {dtype_name}, hdim={head_dim}, kBlockM={kBlockM}, kBlockN={kBlockN}")
            log_print(f"{'=' * 180}")
            log_print(
                f"  {'Profile':>8} {'total_q':>7} {'max_sk':>7} {'sparsity':>8} | "
                f"{'DnV↔DnK':>10} {'SpV↔SpK':>10} {'DnV↔SpV':>10} | "
                f"{'DenseVar':>9} {'SparseVar':>10} {'SparseKV':>9} | "
                f"{'VarSpdUp':>8} {'KV/Var':>7}"
            )
            log_print(f"  {'-' * 165}")

            for pname, profile in PROFILES.items():
                try:
                    r = run_one(dtype, head_dim, pname, profile, warmup=3, iters=15)
                    thr = 0.05 if dtype_name == "fp8" else (0.02 if dtype_name == "bf16" else 0.01)
                    d_ok = "OK" if r["diff_dense"] < thr else "FAIL"
                    s_ok = "OK" if r["diff_sparse"] < thr else "FAIL"
                    a_ok = "OK" if r["rel_approx"] < 0.5 else "WARN"
                    var_speedup = r["dense_var_ms"] / r["sparse_var_ms"] if r["sparse_var_ms"] > 0 else 0.0
                    kv_over_var = r["sparse_kv_ms"] / r["sparse_var_ms"] if r["sparse_var_ms"] > 0 else 0.0

                    log_print(
                        f"  {pname:>8} {r['total_q']:>7} {r['max_sk']:>7} {r['sparsity_pct']:>6.1f}% | "
                        f"{r['diff_dense']:>8.2e}{d_ok:>3} {r['diff_sparse']:>8.2e}{s_ok:>3} "
                        f"{r['rel_approx']:>8.2e}{a_ok:>3} | "
                        f"{r['dense_var_ms']:>7.3f}m {r['sparse_var_ms']:>8.3f}m {r['sparse_kv_ms']:>7.3f}m | "
                        f"{var_speedup:>6.2f}x {kv_over_var:>6.2f}x"
                    )
                except Exception as e:
                    log_print(f"  {pname:>8} | ERROR: {e}")
                    traceback.print_exc()
                finally:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            log_print(f"  {'-' * 165}")

    log_print(f"\n{'=' * 180}")
    log_print("Legend:")
    log_print("  DnV↔DnK : dense flash_attn_varlen_func vs dense flash_attn_with_kvcache(non-paged) max abs diff")
    log_print("  SpV↔SpK : sparse flash_attn_varlen_func vs sparse flash_attn_with_kvcache(non-paged) max abs diff")
    log_print("  DnV↔SpV : dense varlen_func vs sparse varlen_func relative L2 error (approximation)")
    log_print("  DenseVar/SparseVar : flash_attn_varlen_func dense/sparse latency ms")
    log_print("  SparseKV          : flash_attn_with_kvcache(non-paged) sparse latency ms, reference")
    log_print("  VarSpdUp          : DenseVar / SparseVar")
    log_print("  KV/Var            : SparseKV / SparseVar, >1 means varlen_func sparse is faster")
    log_print(f"{'=' * 180}")

    with open(LOG_FILE, "w") as f:
        f.write("\n".join(_log_lines))
    log_print(f"\nLog saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
