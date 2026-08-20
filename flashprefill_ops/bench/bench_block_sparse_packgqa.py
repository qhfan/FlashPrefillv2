"""
Benchmark block_sparse speed: pack_gqa=True vs pack_gqa=False.
Both modes matched to the SAME target sparsity ratio.

Sparsity = selected_KV_blocks / total_causal_KV_blocks * 100

For each seq len, we search (sink, window, last_n, random) separately
for pack_gqa=True and pack_gqa=False to find configs that hit the
same target sparsity. Then we compare efficiency.

Key difference in sparse index construction:
- pack_gqa=True:  total_q_tiles = ceil_div(seqlen_q * gqa_ratio, kBlockM) * batch
                  Q tile covers kBlockM/gqa_ratio q_positions → tighter causal
- pack_gqa=False: total_q_tiles = ceil_div(seqlen_q, kBlockM) * batch
                  Q tile covers kBlockM q_positions → looser causal
                  index organized by Q head (segment g = total_q_tiles * h_q + m_block)
"""

import torch
import numpy as np
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_block_sparse import (
    get_tile_sizes, create_test_inputs,
    m_block_to_q_pos, q_pos_to_k_block,
)
from flash_attn_interface import flash_attn_with_kvcache

torch.manual_seed(42)
device = "cuda"

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"bench_block_sparse_packgqa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

_log_lines = []

def log_print(msg=""):
    print(msg)
    _log_lines.append(msg)


def run_fa3_packgqa(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                    seqlen_q, head_dim, softmax_scale, causal=True,
                    block_sparse_cu=None, block_sparse_idx=None,
                    total_q_tiles=None, cu_q_tiles=None,
                    num_splits=0):
    kwargs = dict(
        q=q, k_cache=k_cache, v_cache=v_cache,
        page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=seqlen_q,
        softmax_scale=softmax_scale, causal=causal,
        num_splits=num_splits,
    )
    if block_sparse_cu is not None:
        kwargs.update(
            block_sparse_cu=block_sparse_cu,
            block_sparse_idx=block_sparse_idx,
            total_q_tiles=total_q_tiles,
            cu_q_tiles=cu_q_tiles,
        )
    return flash_attn_with_kvcache(**kwargs)


def build_index_packgqa(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    attention_sink=1, window=2, last_n_blocks=2, num_random_blocks=0,
    causal=True, device="cuda", rng_seed=42,
):
    """Build sparse index for PackGQA=True mode.
    total_q_tiles = ceil_div(seqlen_q * gqa_ratio, kBlockM) * batch
    Segment: g = total_q_tiles * h_kv + (cu_q_tiles[b] + m_block)
    """
    gqa_ratio = nheads_q // nheads_kv
    n_q_tiles_per_batch = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    n_k_tiles_per_batch = (seqlen_k + kBlockN - 1) // kBlockN
    total_q_tiles = n_q_tiles_per_batch * batch_size

    cu_q_tiles = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + n_q_tiles_per_batch

    rng = np.random.RandomState(rng_seed)
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
                        selected.update(rng.choice(remaining, size=n_sel, replace=False).tolist())

                all_indices.extend(sorted(selected))
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


def build_index_no_packgqa(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    attention_sink=1, window=2, last_n_blocks=2, num_random_blocks=0,
    causal=True, device="cuda", rng_seed=42,
):
    """Build sparse index for PackGQA=False mode.
    total_q_tiles = ceil_div(seqlen_q, kBlockM) * batch
    Index organized by Q head: segment g = total_q_tiles * h_q + m_block
    """
    n_q_tiles_per_batch = (seqlen_q + kBlockM - 1) // kBlockM
    n_k_tiles_per_batch = (seqlen_k + kBlockN - 1) // kBlockN
    total_q_tiles = n_q_tiles_per_batch * batch_size

    cu_q_tiles = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        cu_q_tiles[b + 1] = cu_q_tiles[b] + n_q_tiles_per_batch

    rng = np.random.RandomState(rng_seed)
    n_q_pos_blocks = n_q_tiles_per_batch
    last_n_q_pos_start = max(0, n_q_pos_blocks - last_n_blocks)

    all_indices = []
    cu_offsets = [0]

    for h_q in range(nheads_q):
        for b in range(batch_size):
            prefix_len = seqlen_k - seqlen_q
            for m_block in range(n_q_tiles_per_batch):
                q_pos_start = m_block * kBlockM
                if q_pos_start >= seqlen_q:
                    cu_offsets.append(len(all_indices))
                    continue
                q_k_blk = q_pos_to_k_block(q_pos_start, prefix_len, kBlockN)
                causal_max_n = min(q_k_blk + 1, n_k_tiles_per_batch) if causal else n_k_tiles_per_batch
                is_last_n = m_block >= last_n_q_pos_start

                selected = set()
                selected.update(range(min(attention_sink, causal_max_n)))
                selected.update(range(max(0, q_k_blk - window + 1), min(q_k_blk + 1, causal_max_n)))
                if is_last_n:
                    selected.update(range(causal_max_n))
                else:
                    remaining = [k for k in range(causal_max_n) if k not in selected]
                    if remaining and num_random_blocks > 0:
                        n_sel = min(num_random_blocks, len(remaining))
                        selected.update(rng.choice(remaining, size=n_sel, replace=False).tolist())

                all_indices.extend(sorted(selected))
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


# Target sparsity per seq len (matched to FlashPrefill comparison table)
TARGET_SPARSITY = {
    4096: 70.4,
    8192: 46.0,
    16384: 29.0,
    32768: 17.6,
    65536: 10.0,
}

# Fixed sparse pattern parameters
ATTENTION_SINK = 1
WINDOW = 2
LAST_N_BLOCK = 2


def compute_sparsity_packgqa(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                              kBlockM, kBlockN, sink=ATTENTION_SINK, window=WINDOW, last_n=LAST_N_BLOCK, random_blocks=0):
    """Compute actual sparsity for pack_gqa=True mode."""
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


def compute_sparsity_no_packgqa(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                                 kBlockM, kBlockN, sink=ATTENTION_SINK, window=WINDOW, last_n=LAST_N_BLOCK, random_blocks=0):
    """Compute actual sparsity for pack_gqa=False mode."""
    bs_cu, bs_idx, _, _ = build_index_no_packgqa(
        batch, nheads_q, nheads_kv, seqlen_q, seqlen_k, kBlockM, kBlockN,
        attention_sink=sink, window=window, last_n_blocks=last_n,
        num_random_blocks=random_blocks, causal=True, device="cpu")
    total_selected = bs_idx.shape[0]

    n_q_tiles = (seqlen_q + kBlockM - 1) // kBlockM
    n_k_tiles = (seqlen_k + kBlockN - 1) // kBlockN
    prefix_len = seqlen_k - seqlen_q
    total_causal = 0
    for _ in range(nheads_q):
        for _ in range(batch):
            for m_block in range(n_q_tiles):
                q_pos = m_block * kBlockM
                if q_pos >= seqlen_q:
                    continue
                q_k = q_pos_to_k_block(q_pos, prefix_len, kBlockN)
                total_causal += min(q_k + 1, n_k_tiles)
    if total_causal == 0:
        return 0.0
    return total_selected / total_causal * 100


def find_best_config(mode, batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                     kBlockM, kBlockN, target_sparsity):
    """Search random_blocks closest to target sparsity, with fixed sink=1, window=2, last_n=2."""
    compute_fn = compute_sparsity_packgqa if mode == "pack" else compute_sparsity_no_packgqa
    best = None
    best_diff = 999.0
    for rand in range(0, 33):
        sp = compute_fn(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                        kBlockM, kBlockN, ATTENTION_SINK, WINDOW, LAST_N_BLOCK, rand)
        diff = abs(sp - target_sparsity)
        if diff < best_diff:
            best_diff = diff
            best = (ATTENTION_SINK, WINDOW, LAST_N_BLOCK, rand, sp)
        if diff < 0.5:
            return best
    return best


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


CONCURRENCY = [1, 2, 4, 8]


def main():
    NHEADS_Q = 8
    NHEADS_KV = 2  # GQA ratio = 4

    log_print(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log will be saved to: {LOG_FILE}")

    for head_dim in [128, 256]:
        for dtype, dtype_name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            element_size = 1 if dtype == torch.float8_e4m3fn else 2
            kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

            log_print(f"\n{'='*140}")
            log_print(f"  PackGQA Benchmark (matched sparsity + dense): {dtype_name}, hdim={head_dim}, Q={NHEADS_Q} KV={NHEADS_KV} (ratio={NHEADS_Q//NHEADS_KV}), kBlockM={kBlockM}, kBlockN={kBlockN}")
            log_print(f"{'='*140}")

            # Find matching configs for each seq len (use batch=1 for search)
            configs = {}
            for sq in sorted(TARGET_SPARSITY.keys()):
                target = TARGET_SPARSITY[sq]
                best_pg = find_best_config("pack", 1, NHEADS_Q, NHEADS_KV, sq, sq, kBlockM, kBlockN, target)
                best_np = find_best_config("no_pack", 1, NHEADS_Q, NHEADS_KV, sq, sq, kBlockM, kBlockN, target)

                if best_pg is None or best_np is None:
                    log_print(f"  WARNING: no config found for seq={sq}")
                    continue

                configs[sq] = {"pack": best_pg, "no_pack": best_np, "target": target}
                pg_sink, pg_win, pg_ln, pg_rand, pg_sp = best_pg
                np_sink, np_win, np_ln, np_rand, np_sp = best_np
                log_print(f"  Seq={sq:>5}: target={target:>5.1f}%  "
                      f"pack(s={pg_sink},w={pg_win},ln={pg_ln},r={pg_rand})={pg_sp:>5.1f}%  "
                      f"no_pack(s={np_sink},w={np_win},ln={np_ln},r={np_rand})={np_sp:>5.1f}%")

            for sq in sorted(configs.keys()):
                cfg = configs[sq]
                target = cfg["target"]
                pg_sink, pg_win, pg_ln, pg_rand, pg_sp = cfg["pack"]
                np_sink, np_win, np_ln, np_rand, np_sp = cfg["no_pack"]

                log_print(f"\n  --- Seq={sq}, target={target:.1f}%  pack_actual={pg_sp:.1f}%  no_pack_actual={np_sp:.1f}% ---")
                log_print(f"  {'Conc':>5} | {'Dense':>9} {'Pack ms':>9} {'NoPack ms':>10} {'P/NP':>7} {'P/D':>7} {'NP/D':>7} {'Pack idx':>9} {'NoPack idx':>10}")
                log_print(f"  {'-'*95}")

                for conc in CONCURRENCY:
                    try:
                        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
                            conc, sq, sq, NHEADS_Q, NHEADS_KV, head_dim, dtype, device)
                        if dtype == torch.float8_e4m3fn:
                            q = q.to(torch.float8_e4m3fn)
                        softmax_scale = head_dim ** (-0.5)

                        # Dense baseline (no sparse, pack_gqa auto)
                        fn_dense = lambda: run_fa3_packgqa(
                            q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                            sq, head_dim, softmax_scale, causal=True)

                        # Build indices for this batch size
                        bs_cu_pg, bs_idx_pg, tqt_pg, cqt_pg = build_index_packgqa(
                            conc, NHEADS_Q, NHEADS_KV, sq, sq, kBlockM, kBlockN,
                            attention_sink=pg_sink, window=pg_win, last_n_blocks=pg_ln,
                            num_random_blocks=pg_rand, device=device)

                        bs_cu_np, bs_idx_np, tqt_np, cqt_np = build_index_no_packgqa(
                            conc, NHEADS_Q, NHEADS_KV, sq, sq, kBlockM, kBlockN,
                            attention_sink=np_sink, window=np_win, last_n_blocks=np_ln,
                            num_random_blocks=np_rand, device=device)

                        fn_pg = lambda: run_fa3_packgqa(
                            q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                            sq, head_dim, softmax_scale, causal=True,
                            block_sparse_cu=bs_cu_pg, block_sparse_idx=bs_idx_pg,
                            total_q_tiles=tqt_pg, cu_q_tiles=cqt_pg)

                        fn_np = lambda: run_fa3_packgqa(
                            q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                            sq, head_dim, softmax_scale, causal=True,
                            block_sparse_cu=bs_cu_np, block_sparse_idx=bs_idx_np,
                            total_q_tiles=tqt_np, cu_q_tiles=cqt_np)

                        med_dense = benchmark_fn(fn_dense, warmup=3, iters=15)
                        med_pg = benchmark_fn(fn_pg, warmup=3, iters=15)
                        med_np = benchmark_fn(fn_np, warmup=3, iters=15)
                        speedup_pn = med_np / med_pg if med_pg > 0 else 0
                        speedup_pd = med_dense / med_pg if med_pg > 0 else 0
                        speedup_npd = med_dense / med_np if med_np > 0 else 0

                        log_print(f"  {conc:>5} | {med_dense*1000:>8.3f} {med_pg*1000:>8.3f} {med_np*1000:>9.3f} "
                              f"{speedup_pn:>6.2f}x {speedup_pd:>6.2f}x {speedup_npd:>6.2f}x "
                              f"{bs_idx_pg.shape[0]:>8} {bs_idx_np.shape[0]:>9}")
                    except Exception as e:
                        import traceback
                        log_print(f"  {conc:>5} | ERROR: {str(e)[:80]}")
                    torch.cuda.empty_cache()
                log_print(f"  {'-'*95}")

    log_print(f"\n{'='*140}")
    log_print("Notes:")
    log_print("  - Sparsity matched: both modes search (sink=1, window=2, last_n=2, random) to hit same target")
    log_print(f"  - GQA: Q={NHEADS_Q} KV={NHEADS_KV} (ratio={NHEADS_Q//NHEADS_KV}), causal=True, last_n={LAST_N_BLOCK}")
    log_print("  - Fixed: attention_sink=1, window=2, last_n_block=2")
    log_print("  - P/NP = NoPack/Pack speedup (>1.0x means Pack is faster)")
    log_print("  - P/D  = Dense/Pack speedup (sparse acceleration ratio)")
    log_print("  - NP/D = Dense/NoPack speedup (sparse acceleration ratio)")
    log_print(f"{'='*140}")

    # Save log
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(_log_lines))
    log_print(f"\nLog saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
