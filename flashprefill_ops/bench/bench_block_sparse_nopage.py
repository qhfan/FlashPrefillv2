"""
Varlen block sparse attention benchmark: non-paged (TMA) vs paged.

Varlen mode: multiple variable-length requests packed together with cu_seqlens_q.
Non-paged: k_cache = (batch, max_seqlen_k, nheads_kv, head_dim), page_table=None → TMA.
Paged:     k_cache = (num_pages, page_size, nheads_kv, head_dim), with page_table.

Correctness:
  1. Dense paged vs dense non-paged (should match)
  2. Sparse paged vs sparse non-paged (should match)
  3. Dense non-paged vs sparse non-paged (approximation quality)
"""
import torch
import numpy as np
import sys, os, time, traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)
import flashprefill
sys.path.insert(0, script_dir)
from test_block_sparse import (
    get_tile_sizes, m_block_to_q_pos, q_pos_to_k_block,
)
from flash_attn_interface import flash_attn_with_kvcache

torch.manual_seed(42)
device = "cuda"

ATTENTION_SINK = 1
WINDOW = 2
LAST_N_BLOCK = 2
RANDOM_BLOCKS = 4

NHEADS_Q = 32
NHEADS_KV = 8
GQA_RATIO = NHEADS_Q // NHEADS_KV

# Varlen profiles: list of (seqlen_q, seqlen_k) per request.
# seqlen_k = prefix_len + seqlen_q (KV cache has prefix + current prefill chunk)
PROFILES = {
    "short":  [(1024, 4096), (2048, 8192), (1536, 6144), (3072, 12288)],
    "medium": [(4096, 16384), (2048, 12288), (6144, 24576), (3072, 16384)],
    "long":   [(8192, 32768), (4096, 28672), (6144, 32768), (3072, 24576)],
    "xlong":  [(8192, 65536), (4096, 49152), (12288, 65536), (6144, 49152)],
}


# ──────────────────────────────────────────────────────────────────────────────
#  Input creation
# ──────────────────────────────────────────────────────────────────────────────

def create_varlen_inputs(profile, nheads_q, nheads_kv, head_dim, dtype, page_size=1):
    """
    Create varlen inputs for both non-paged and paged modes (same data).

    Returns:
      q:                (total_q, nheads_q, head_dim)
      k_nopage:         (batch, max_seqlen_k, nheads_kv, head_dim)
      v_nopage:         (batch, max_seqlen_k, nheads_kv, head_dim)
      k_paged:          (num_pages, page_size, nheads_kv, head_dim)
      v_paged:          (num_pages, page_size, nheads_kv, head_dim)
      page_table:       (batch, max_seqlen_k) int32
      cache_seqlens:    (batch,) int32 — actual K length per batch
      cu_seqlens_q:     (batch+1,) int32
      seqlens_q:        list[int]
      seqlens_k:        list[int]
    """
    batch = len(profile)
    seqlens_q = [p[0] for p in profile]
    seqlens_k = [p[1] for p in profile]
    total_q = sum(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    scale = 0.5

    # Q: packed (total_q, nheads_q, head_dim)
    if dtype == torch.float8_e4m3fn:
        q = (torch.randn(total_q, nheads_q, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)
    else:
        q = torch.randn(total_q, nheads_q, head_dim, dtype=dtype, device=device) * scale

    # Non-paged K/V: (batch, max_seqlen_k, nheads_kv, head_dim)
    if dtype == torch.float8_e4m3fn:
        k_nopage = (torch.randn(batch, max_seqlen_k, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)
        v_nopage = (torch.randn(batch, max_seqlen_k, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale).to(dtype)
    else:
        k_nopage = torch.randn(batch, max_seqlen_k, nheads_kv, head_dim, dtype=dtype, device=device) * scale
        v_nopage = torch.randn(batch, max_seqlen_k, nheads_kv, head_dim, dtype=dtype, device=device) * scale

    # Paged K/V: (num_pages, page_size, nheads_kv, head_dim) — same data
    num_pages = batch * max_seqlen_k
    k_paged = k_nopage.reshape(num_pages, page_size, nheads_kv, head_dim).contiguous()
    v_paged = v_nopage.reshape(num_pages, page_size, nheads_kv, head_dim).contiguous()

    # page_table: sequential mapping (batch b, token t) → page b*max_seqlen_k + t
    page_table = torch.zeros(batch, max_seqlen_k, dtype=torch.int32, device=device)
    for b in range(batch):
        page_table[b] = torch.arange(b * max_seqlen_k, (b + 1) * max_seqlen_k, dtype=torch.int32, device=device)

    # cu_seqlens_q
    cu_seqlens_q = torch.tensor([0] + np.cumsum(seqlens_q).tolist(), dtype=torch.int32, device=device)

    # cache_seqlens
    cache_seqlens = torch.tensor(seqlens_k, dtype=torch.int32, device=device)

    return (q, k_nopage, v_nopage, k_paged, v_paged, page_table,
            cache_seqlens, cu_seqlens_q, seqlens_q, seqlens_k, max_seqlen_k)


# ──────────────────────────────────────────────────────────────────────────────
#  Sparse index builder (varlen-aware)
# ──────────────────────────────────────────────────────────────────────────────

def build_varlen_sparse_index(
    seqlens_q, seqlens_k, nheads_q, nheads_kv,
    kBlockM, kBlockN,
    attention_sink=ATTENTION_SINK, window=WINDOW, last_n_blocks=LAST_N_BLOCK,
    num_random_blocks=RANDOM_BLOCKS, causal=True, device="cuda", rng_seed=42,
):
    """
    Build CSR sparse index for varlen packed Q.

    Layout: for each h_kv, for each batch b, for each q_tile in b:
      segment_id = total_q_tiles_so_far + m_block
    """
    gqa_ratio = nheads_q // nheads_kv
    batch = len(seqlens_q)

    # Compute q_tiles per batch (in GQA tile units)
    q_tiles_per_batch = []
    for b in range(batch):
        sq = seqlens_q[b]
        n_tiles = (sq * gqa_ratio + kBlockM - 1) // kBlockM
        q_tiles_per_batch.append(n_tiles)
    total_q_tiles = sum(q_tiles_per_batch) * nheads_kv

    # cu_q_tiles: prefix sum over (batch * nheads_kv) — but layout is h_kv major
    # In the kernel: g = total_q_tiles * bidh_kv + global_m_block
    # global_m_block = cu_q_tiles[bidb] + m_block
    # So cu_q_tiles is per-batch (not per-h_kv), and total_q_tiles = sum(q_tiles_per_batch)
    total_q_tiles = sum(q_tiles_per_batch)
    cu_q_tiles = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    for b in range(batch):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + q_tiles_per_batch[b]

    rng = np.random.RandomState(rng_seed)
    positions_per_m_block = max(1, kBlockM // gqa_ratio)

    all_indices = []
    cu_offsets = [0]

    for h_kv in range(nheads_kv):
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
                    remaining = [k for k in range(causal_max_n) if k not in selected]
                    if remaining and num_random_blocks > 0:
                        n_sel = min(num_random_blocks, len(remaining))
                        pick_indices = np.linspace(0, len(remaining) - 1, n_sel, dtype=int)
                        selected.update(remaining[i] for i in pick_indices)

                all_indices.extend(sorted(selected))
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


# ──────────────────────────────────────────────────────────────────────────────
#  FA3 wrapper
# ──────────────────────────────────────────────────────────────────────────────

def run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
            max_seqlen_q, head_dim, softmax_scale, causal=True,
            block_sparse_cu=None, block_sparse_idx=None,
            total_q_tiles=None, cu_q_tiles=None, sinks=None):
    kwargs = dict(
        q=q, k_cache=k_cache, v_cache=v_cache,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale, causal=causal,
    )
    if block_sparse_cu is not None:
        kwargs.update(
            block_sparse_cu=block_sparse_cu,
            block_sparse_idx=block_sparse_idx,
            total_q_tiles=total_q_tiles,
            cu_q_tiles=cu_q_tiles,
        )
    if sinks is not None:
        kwargs["sinks"] = sinks
    return flash_attn_with_kvcache(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
#  Utils
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def run_one(dtype, head_dim, profile_name, profile, warmup=3, iters=15):
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=False)
    softmax_scale = head_dim ** (-0.5)
    dtype_name = "fp8" if dtype == torch.float8_e4m3fn else ("fp16" if dtype == torch.float16 else "bf16")
    max_seqlen_q = max(p[0] for p in profile)

    # Create inputs
    (q, k_nopage, v_nopage, k_paged, v_paged, page_table,
     cache_seqlens, cu_seqlens_q, seqlens_q, seqlens_k, max_seqlen_k) = create_varlen_inputs(
        profile, NHEADS_Q, NHEADS_KV, head_dim, dtype)

    sinks = torch.zeros(NHEADS_Q, dtype=q.dtype, device=device)

    # Build sparse index
    bs_cu, bs_idx, total_q_tiles, cu_q_tiles = build_varlen_sparse_index(
        seqlens_q, seqlens_k, NHEADS_Q, NHEADS_KV,
        kBlockM, kBlockN, device=device)

    # ── Correctness ──
    out_dense_paged = run_fa3(q, k_paged, v_paged, page_table, cache_seqlens, cu_seqlens_q,
                              max_seqlen_q, head_dim, softmax_scale, causal=True, sinks=sinks)
    out_dense_nopage = run_fa3(q, k_nopage, v_nopage, None, cache_seqlens, cu_seqlens_q,
                               max_seqlen_q, head_dim, softmax_scale, causal=True, sinks=sinks)
    diff_dense = max_abs_diff(out_dense_paged, out_dense_nopage)

    out_sparse_paged = run_fa3(q, k_paged, v_paged, page_table, cache_seqlens, cu_seqlens_q,
                               max_seqlen_q, head_dim, softmax_scale, causal=True,
                               block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                               total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles, sinks=sinks)
    out_sparse_nopage = run_fa3(q, k_nopage, v_nopage, None, cache_seqlens, cu_seqlens_q,
                                max_seqlen_q, head_dim, softmax_scale, causal=True,
                                block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                                total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles, sinks=sinks)
    diff_sparse = max_abs_diff(out_sparse_paged, out_sparse_nopage)
    rel_approx = rel_error(out_dense_nopage, out_sparse_nopage)

    # ── Benchmark ──
    fn_dense_paged = lambda: run_fa3(q, k_paged, v_paged, page_table, cache_seqlens, cu_seqlens_q,
                                     max_seqlen_q, head_dim, softmax_scale, causal=True, sinks=sinks)
    fn_dense_nopage = lambda: run_fa3(q, k_nopage, v_nopage, None, cache_seqlens, cu_seqlens_q,
                                      max_seqlen_q, head_dim, softmax_scale, causal=True, sinks=sinks)
    fn_sparse_paged = lambda: run_fa3(q, k_paged, v_paged, page_table, cache_seqlens, cu_seqlens_q,
                                      max_seqlen_q, head_dim, softmax_scale, causal=True,
                                      block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                                      total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles, sinks=sinks)
    fn_sparse_nopage = lambda: run_fa3(q, k_nopage, v_nopage, None, cache_seqlens, cu_seqlens_q,
                                       max_seqlen_q, head_dim, softmax_scale, causal=True,
                                       block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                                       total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles, sinks=sinks)

    med = {k: benchmark_fn(fn, warmup, iters) * 1000 for k, fn in [
        ("dense_pg", fn_dense_paged), ("dense_np", fn_dense_nopage),
        ("sparse_pg", fn_sparse_paged), ("sparse_np", fn_sparse_nopage),
    ]}

    # Sparsity
    total_causal = 0
    total_selected = bs_idx.shape[0]
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
    sparsity_pct = total_selected / total_causal * 100 if total_causal > 0 else 0

    return {
        "dtype": dtype_name, "hdim": head_dim, "profile": profile_name,
        "kBlockM": kBlockM, "kBlockN": kBlockN,
        "total_q": sum(seqlens_q), "max_sk": max_seqlen_k,
        "sparsity_pct": sparsity_pct,
        "diff_dense": diff_dense,
        "diff_sparse": diff_sparse,
        "rel_approx": rel_approx,
        **med,
    }


def main():
    print(f"\n{'='*200}")
    print(f"  Varlen Block Sparse Attention — Non-Paged (TMA) vs Paged Benchmark")
    print(f"  GQA: Q={NHEADS_Q} KV={NHEADS_KV} (ratio={GQA_RATIO}), causal=True")
    print(f"  Sparse: sink={ATTENTION_SINK}, window={WINDOW}, last_n={LAST_N_BLOCK}, random={RANDOM_BLOCKS}")
    print(f"  Non-paged: page_table=None → PagedKVNonTMA=false → Use_TMA_KV=true")
    print(f"{'='*200}")

    for head_dim in [128, 256]:
        for dtype, dtype_name in [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            element_size = 1 if dtype == torch.float8_e4m3fn else 2
            kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=False)

            print(f"\n{'─'*200}")
            print(f"  {dtype_name}, hdim={head_dim}, kBlockM={kBlockM}, kBlockN={kBlockN}")
            print(f"{'─'*200}")

            print(f"\n  {'Profile':>8} {'total_q':>7} {'max_sk':>7} {'sparsity':>8} | "
                  f"{'DnPg↔DnNp':>10} {'SpPg↔SpNp':>10} {'DnNp↔SpNp':>10} | "
                  f"{'DensePg':>9} {'DenseNp':>9} {'SparsPg':>9} {'SparsNp':>9} | "
                  f"{'NpSpdUp':>8} {'Pg/Np':>7}")
            print(f"  {'─'*170}")

            for pname, profile in PROFILES.items():
                try:
                    r = run_one(dtype, head_dim, pname, profile, warmup=3, iters=15)

                    thr = 0.05 if dtype_name == "fp8" else 0.01
                    d_ok = "OK" if r["diff_dense"] < thr else "FAIL"
                    s_ok = "OK" if r["diff_sparse"] < thr else "FAIL"
                    a_ok = "OK" if r["rel_approx"] < 0.5 else "WARN"

                    np_speedup = r["dense_np"] / r["sparse_np"] if r["sparse_np"] > 0 else 0
                    pg_np = r["sparse_pg"] / r["sparse_np"] if r["sparse_np"] > 0 else 0

                    print(
                        f"  {pname:>8} {r['total_q']:>7} {r['max_sk']:>7} {r['sparsity_pct']:>6.1f}% | "
                        f"{r['diff_dense']:>8.2e}{d_ok:>3} {r['diff_sparse']:>8.2e}{s_ok:>3} "
                        f"{r['rel_approx']:>8.2e}{a_ok:>3} | "
                        f"{r['dense_pg']:>7.3f}m {r['dense_np']:>7.3f}m "
                        f"{r['sparse_pg']:>7.3f}m {r['sparse_np']:>7.3f}m | "
                        f"{np_speedup:>6.2f}x {pg_np:>6.2f}x"
                    )
                except Exception as e:
                    print(f"  {pname:>8} | ERROR: {e}")
                    traceback.print_exc()
                finally:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    print(f"\n{'='*200}")
    print("Legend:")
    print("  DnPg↔DnNp : dense paged vs dense non-paged max abs diff (should be ~0)")
    print("  SpPg↔SpNp : sparse paged vs sparse non-paged max abs diff (should be ~0)")
    print("  DnNp↔SpNp : dense non-paged vs sparse non-paged rel L2 error (approximation)")
    print("  DensePg/DenseNp : dense latency (paged / non-paged) ms")
    print("  SparsPg/SparsNp: sparse latency (paged / non-paged) ms")
    print("  NpSpdUp   : non-paged sparse speedup = dense_np / sparse_np")
    print("  Pg/Np     : paged vs non-paged sparse speedup = sparse_pg / sparse_np")
    print(f"{'='*200}")


if __name__ == "__main__":
    main()
