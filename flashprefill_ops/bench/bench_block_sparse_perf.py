"""Block sparse attention performance comparison (all indices prebuilt outside the timed region).

dense  = stock FA3 from the environment (site-packages flash_attn_interface.py, loaded via importlib absolute path)
sparse = this repo's flashprefill implementation (block sparse path derived from FA3)

Three scenarios:
  A) fwd varlen prefill
  B) bwd varlen (direct call to the low-level _flash_attn_backward; reverse index prebuilt)
  C) with_kv_cache (paged KV fwd)

Output: sparsity (Sparse%), theoretical FLOPs ratio (FLOPs_R), actual speedup (Speedup).
"""

import importlib.util
import os
import sys
import time

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)
import flashprefill  # noqa: F401  (registers the torch ops of flashprefill._C)
sys.path.insert(0, script_dir)

from flash_attn_interface import (  # noqa: E402  (this repo's implementation, sparse side)
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    _flash_attn_backward,
)
from flash_block_sparse_bwd_index import build_block_sparse_bwd_index  # noqa: E402
from test_block_sparse import (  # noqa: E402
    get_tile_sizes,
    create_test_inputs,
    m_block_to_q_pos,
    q_pos_to_k_block,
)

# Stock FA3 (dense baseline), loaded via importlib absolute path to avoid being shadowed by the repo's same-named module
_spec = importlib.util.spec_from_file_location(
    "fa3_orig_interface", "/usr/local/lib/python3.12/dist-packages/flash_attn_interface.py")
fa3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa3)
assert ("site-packages" in fa3.__file__ or "dist-packages" in fa3.__file__), fa3.__file__

torch.manual_seed(42)
device = "cuda"

# ---------------- Configuration ----------------
NHEADS_Q = 32
NHEADS_KV = 8          # GQA ratio = 4
BATCH = 1
SEQ_LENS = [4096, 8192, 16384, 32768]
FLASH_PREFILL_TARGETS = {4096: 70.4, 8192: 46.0, 16384: 29.0, 32768: 17.6}
ATTENTION_SINK = 0
WINDOW = 0
LAST_N_BLOCK = 0


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


# ---------------- Sparse index construction (deterministic, mask generated directly; not timed) ----------------

def build_index_packgqa(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    attention_sink=ATTENTION_SINK, window=WINDOW, last_n_blocks=LAST_N_BLOCK,
    num_random_blocks=0, causal=True, device="cuda", rng_seed=42,
):
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles_per_batch = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    n_k_tiles_per_batch = (seqlen_k + kBlockN - 1) // kBlockN
    total_q_tiles = n_q_tiles_per_batch * batch_size

    cu_q_tiles = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + n_q_tiles_per_batch

    positions_per_m_block = max(1, kBlockM // gqa_ratio)
    n_q_pos_blocks = (seqlen_q + positions_per_m_block - 1) // positions_per_m_block
    last_n_q_pos_start = max(0, n_q_pos_blocks - last_n_blocks)

    all_indices = []
    cu_offsets = [0]
    for h_kv in range(nheads_kv):
        for b in range(batch_size):
            prefix_len = seqlen_k - seqlen_q
            for m_block in range(n_q_tiles_per_batch):
                q_pos_start = m_block_to_q_pos(m_block, kBlockM, gqa_ratio)
                if q_pos_start >= seqlen_q:
                    cu_offsets.append(len(all_indices))
                    continue
                q_k_blk = q_pos_to_k_block(q_pos_start, prefix_len, kBlockN)
                causal_max_n = min(q_k_blk + 1, n_k_tiles_per_batch) if causal else n_k_tiles_per_batch
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


def compute_sparsity(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                     kBlockM, kBlockN, sink, window, last_n, random_blocks):
    bs_cu, bs_idx, _, _ = build_index_packgqa(
        batch, nheads_q, nheads_kv, seqlen_q, seqlen_k, kBlockM, kBlockN,
        attention_sink=sink, window=window, last_n_blocks=last_n,
        num_random_blocks=random_blocks, causal=True, device="cpu")
    total_selected = bs_idx.shape[0]
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    n_k_tiles = (seqlen_k + kBlockN - 1) // kBlockN
    prefix_len = seqlen_k - seqlen_q
    total_causal = 0
    for _ in range(nheads_kv):
        for _ in range(batch):
            for m_block in range(n_q_tiles):
                q_pos = m_block_to_q_pos(m_block, kBlockM, gqa_ratio)
                if q_pos >= seqlen_q:
                    continue
                q_k = q_pos_to_k_block(q_pos, prefix_len, kBlockN)
                total_causal += min(q_k + 1, n_k_tiles)
    if total_causal == 0:
        return 0.0
    return total_selected / total_causal * 100


def find_best_config(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                     kBlockM, kBlockN, target_sparsity):
    """Sparsity is monotonically non-decreasing in random_blocks; binary search."""
    n_k_max = (seqlen_k + kBlockN - 1) // kBlockN
    lo, hi = 0, n_k_max
    best = None
    best_diff = 999.0
    while lo <= hi:
        mid = (lo + hi) // 2
        sp = compute_sparsity(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                              kBlockM, kBlockN, ATTENTION_SINK, WINDOW, LAST_N_BLOCK, mid)
        diff = sp - target_sparsity
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best = (ATTENTION_SINK, WINDOW, LAST_N_BLOCK, mid, sp)
        if abs(diff) < 0.5:
            return best
        if diff < 0:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ---------------- Scenarios ----------------

def make_qkv(sq, head_dim, dtype):
    """Create tensors via randn directly in dtype; fp8 must be converted from fp16 first."""
    if dtype == torch.float8_e4m3fn:
        q = (torch.randn(sq, NHEADS_Q, head_dim, dtype=torch.float16, device=device) * 0.5).to(dtype)
        k = (torch.randn(sq, NHEADS_KV, head_dim, dtype=torch.float16, device=device) * 0.5).to(dtype)
        v = (torch.randn(sq, NHEADS_KV, head_dim, dtype=torch.float16, device=device) * 0.5).to(dtype)
    else:
        q = torch.randn(sq, NHEADS_Q, head_dim, dtype=dtype, device=device) * 0.5
        k = torch.randn(sq, NHEADS_KV, head_dim, dtype=dtype, device=device) * 0.5
        v = torch.randn(sq, NHEADS_KV, head_dim, dtype=dtype, device=device) * 0.5
    return q, k, v


def run_fwd_varlen(head_dim, dtype, sq, rand_blocks):
    """Scenario A: fwd varlen prefill, batch=1 single sequence."""
    kBlockM, kBlockN = 128, 64
    scale = head_dim ** -0.5
    q, k, v = make_qkv(sq, head_dim, dtype)
    cu = torch.tensor([0, sq], dtype=torch.int32, device=device)

    bs_cu, bs_idx, total_q_tiles, cu_q_tiles = build_index_packgqa(
        BATCH, NHEADS_Q, NHEADS_KV, sq, sq, kBlockM, kBlockN,
        num_random_blocks=rand_blocks, causal=True, device=device)

    fn_dense = lambda: fa3.flash_attn_varlen_func(
        q, k, v, cu, cu, sq, sq, causal=True)
    fn_sparse = lambda: flash_attn_varlen_func(
        q, k, v, cu, cu, sq, sq, causal=True,
        block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
        total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)

    t_dense = benchmark_fn(fn_dense)
    t_sparse = benchmark_fn(fn_sparse)
    return t_dense, t_sparse, bs_idx.shape[0]


def run_bwd_varlen(head_dim, dtype, sq, rand_blocks):
    """Scenario B: bwd varlen, direct call to the low-level _flash_attn_backward, index prebuilt."""
    scale = head_dim ** -0.5
    q = torch.randn(sq, NHEADS_Q, head_dim, dtype=dtype, device=device) * 0.5
    k = torch.randn(sq, NHEADS_KV, head_dim, dtype=dtype, device=device) * 0.5
    v = torch.randn(sq, NHEADS_KV, head_dim, dtype=dtype, device=device) * 0.5
    dout = torch.randn(sq, NHEADS_Q, head_dim, dtype=dtype, device=device) * 0.5
    cu = torch.tensor([0, sq], dtype=torch.int32, device=device)

    bs_cu, bs_idx, total_q_tiles, cu_q_tiles = build_index_packgqa(
        BATCH, NHEADS_Q, NHEADS_KV, sq, sq, 128, 64,
        num_random_blocks=rand_blocks, causal=True, device=device)
    bwd_cu, bwd_idx, max_k_tiles = build_block_sparse_bwd_index(
        bs_cu, bs_idx, cu_q_tiles, cu, cu, NHEADS_Q, NHEADS_KV, total_q_tiles, sq)

    # Pre-run fwd to get out/lse (each path uses its own forward)
    out_d, lse_d = fa3.flash_attn_varlen_func(
        q, k, v, cu, cu, sq, sq, causal=True, return_attn_probs=True)[:2]
    out_s, lse_s = flash_attn_varlen_func(
        q, k, v, cu, cu, sq, sq, causal=True,
        block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
        total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles,
        return_attn_probs=True)[:2]

    dq_d, dk_d, dv_d = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dq_s, dk_s, dv_s = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    fn_dense = lambda: fa3._flash_attn_backward(
        dout, q, k, v, out_d, lse_d, cu, cu, None, None, sq, sq,
        dq_d, dk_d, dv_d, scale, True, (-1, -1), 0.0, False, 0)
    fn_sparse = lambda: _flash_attn_backward(
        dout, q, k, v, out_s, lse_s, cu, cu, None, None, sq, sq,
        dq_s, dk_s, dv_s, scale, True, -1, -1, 0.0, False, 0,
        bwd_cu, bwd_idx, max_k_tiles)

    t_dense = benchmark_fn(fn_dense)
    t_sparse = benchmark_fn(fn_sparse)
    return t_dense, t_sparse, bs_idx.shape[0]


def run_kvcache(head_dim, dtype, sq, rand_blocks):
    """Scenario C: with_kv_cache (paged KV fwd), q_len == kv_len."""
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)
    scale = head_dim ** -0.5
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        BATCH, sq, sq, NHEADS_Q, NHEADS_KV, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(dtype)

    bs_cu, bs_idx, total_q_tiles, cu_q_tiles = build_index_packgqa(
        BATCH, NHEADS_Q, NHEADS_KV, sq, sq, kBlockM, kBlockN,
        num_random_blocks=rand_blocks, causal=True, device=device)

    fn_dense = lambda: fa3.flash_attn_with_kvcache(
        q=q, k_cache=k_cache, v_cache=v_cache,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=sq,
        softmax_scale=scale, causal=True, num_splits=1)
    fn_sparse = lambda: flash_attn_with_kvcache(
        q=q, k_cache=k_cache, v_cache=v_cache,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=sq,
        softmax_scale=scale, causal=True, num_splits=1,
        block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
        total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)

    t_dense = benchmark_fn(fn_dense)
    t_sparse = benchmark_fn(fn_sparse)
    return t_dense, t_sparse, bs_idx.shape[0]


def causal_total(sq, rand_blocks):
    """Theoretical FLOPs ratio under this pattern (total_causal / total_selected)."""
    gqa_ratio = NHEADS_Q // NHEADS_KV
    n_q_tiles = (sq * gqa_ratio + 127) // 128
    n_k_tiles = (sq + 63) // 64
    bs_cu, bs_idx, _, _ = build_index_packgqa(
        BATCH, NHEADS_Q, NHEADS_KV, sq, sq, 128, 64,
        num_random_blocks=rand_blocks, causal=True, device="cpu")
    total_selected = bs_idx.shape[0]
    total = 0
    for _ in range(NHEADS_KV):
        for m_block in range(n_q_tiles):
            q_pos = m_block_to_q_pos(m_block, 128, gqa_ratio)
            if q_pos >= sq:
                continue
            q_k = q_pos_to_k_block(q_pos, 0, 64)
            total += min(q_k + 1, n_k_tiles)
    sparsity_pct = total_selected / total * 100
    flops_r = total / total_selected
    return sparsity_pct, flops_r


SCENARIOS = [
    ("A fwd varlen", run_fwd_varlen),
    ("B bwd varlen", run_bwd_varlen),
    ("C with_kv_cache", run_kvcache),
]
_only = os.getenv("BENCH_ONLY", "")
if _only:
    SCENARIOS = [s for s in SCENARIOS if s[0].startswith(_only)]


def main():
    print(f"device: {torch.cuda.get_device_name(0)}, GQA Q={NHEADS_Q} KV={NHEADS_KV} "
          f"(ratio={NHEADS_Q // NHEADS_KV}), batch={BATCH}, bf16")
    print(f"pattern: sink={ATTENTION_SINK}, window={WINDOW}, last_n={LAST_N_BLOCK} + random_blocks "
          f"(matched to target sparsity); all indices prebuilt, not timed")

    if os.getenv("BENCH_SWEEP", ""):
        # Extreme sparsity sweep: BENCH_SWEEP="A B" (scenarios), BENCH_SPS="5,25,50,75,95"
        sweep_scenarios = os.getenv("BENCH_SWEEP", "A B").split()
        sps = [float(x) for x in os.getenv("BENCH_SPS", "5,25,50,75,95").split(",")]
        scene_map = {"A": ("A fwd varlen", run_fwd_varlen), "B": ("B bwd varlen", run_bwd_varlen),
                     "C": ("C with_kv_cache", run_kvcache)}
        for head_dim in [128, 256]:
            for key in sweep_scenarios:
                name, fn = scene_map[key]
                print(f"\n{'=' * 118}")
                print(f"  Scenario {name} | bf16 hdim={head_dim} | sparsity sweep")
                print(f"{'=' * 118}")
                hdr = (f"  {'SeqLen':>7} {'Target%':>8} {'Sparse%':>8} {'FLOPs_R':>8} | "
                       f"{'Dense ms':>9} {'Sparse ms':>10} | {'Speedup':>8}")
                print(hdr)
                print(f"  {'-' * 82}")
                for sq in SEQ_LENS:
                    for target in sps:
                        best = find_best_config(BATCH, NHEADS_Q, NHEADS_KV, sq, sq, 128, 64, target)
                        rand = best[3]
                        sp_pct, flops_r = causal_total(sq, rand)
                        try:
                            t_dense, t_sparse, _ = fn(head_dim, torch.bfloat16, sq, rand)
                            print(f"  {sq:>7} {target:>7.1f}% {sp_pct:>7.1f}% {flops_r:>7.2f}x | "
                                  f"{t_dense * 1000:>9.3f} {t_sparse * 1000:>10.3f} | "
                                  f"{t_dense / t_sparse:>7.2f}x")
                        except Exception as e:
                            print(f"  {sq:>7} {target:>7.1f}% {sp_pct:>7.1f}% {flops_r:>7.2f}x | "
                                  f"ERROR: {type(e).__name__}: {e}")
                        finally:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
        return

    for head_dim in [128, 256]:
        # Determine random_blocks for each seqlen first (independent of hdim, computed only once)
        configs = {}
        for sq in SEQ_LENS:
            best = find_best_config(BATCH, NHEADS_Q, NHEADS_KV, sq, sq, 128, 64,
                                    FLASH_PREFILL_TARGETS[sq])
            configs[sq] = best[3]
        for name, fn in SCENARIOS:
            # FA3's fp8 only has forward; fp8 is not tested in the bwd scenario
            dtype_list = [(torch.bfloat16, "bf16")] if name.startswith("B") else [
                (torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]
            for dtype, dtype_name in dtype_list:
                print(f"\n{'=' * 118}")
                print(f"  Scenario {name} | {dtype_name} hdim={head_dim}")
                print(f"{'=' * 118}")
                hdr = (f"  {'SeqLen':>7} {'Sparse%':>8} {'FLOPs_R':>8} | "
                       f"{'Dense ms':>9} {'Sparse ms':>10} | {'Speedup':>8}")
                print(hdr)
                print(f"  {'-' * 70}")
                for sq in SEQ_LENS:
                    rand = configs[sq]
                    sp_pct, flops_r = causal_total(sq, rand)
                    try:
                        t_dense, t_sparse, _ = fn(head_dim, dtype, sq, rand)
                        print(f"  {sq:>7} {sp_pct:>7.1f}% {flops_r:>7.2f}x | "
                              f"{t_dense * 1000:>9.3f} {t_sparse * 1000:>10.3f} | "
                              f"{t_dense / t_sparse:>7.2f}x")
                    except Exception as e:
                        print(f"  {sq:>7} {sp_pct:>7.1f}% {flops_r:>7.2f}x | ERROR: {type(e).__name__}: {e}")
                    finally:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
    print(f"\n{'=' * 118}")
    print("Notes:")
    print("  - dense = stock FA3 (site-packages flash_attn_interface.py); sparse = this repo's flashprefill")
    print("  - Sparse% = selected K-blocks / total causal-visible K-blocks; FLOPs_R = theoretical speedup upper bound")
    print("  - All indices (fwd CSR + bwd reverse index) are prebuilt outside the loop, not timed")
    print("  - Timing = median of 15 iters (3 warmup), batch=1 single-sequence varlen; kv cache scenario q_len == kv_len")


if __name__ == "__main__":
    main()
