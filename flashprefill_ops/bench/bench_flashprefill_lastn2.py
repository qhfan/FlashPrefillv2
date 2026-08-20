"""
Benchmark the current FlashPrefill public API with matched sparsity levels,
last_n=2, and varying request concurrency.

The static sparse-attention measurement uses a deterministic index whose
selected density matches the comparison-table targets. Index-select
measurement uses ``FlashPrefill.index_select``.
"""
import torch
import numpy as np
import sys, os, time, traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
while script_dir in sys.path:
    sys.path.remove(script_dir)
import flashprefill  # Load the installed package containing flashprefill._C.
from flashprefill import FlashPrefill
sys.path.insert(0, script_dir)
from test_block_sparse import (
    get_tile_sizes, create_test_inputs,
    m_block_to_q_pos, q_pos_to_k_block,
)
from flash_attn_interface import flash_attn_with_kvcache
from flash_attn.cute.interface import flash_attn_varlen_func as fa_cute_varlen  # official dense baseline
from flash_attn import flash_attn_varlen_func as fa2_varlen  # FA2 dense baseline

torch.manual_seed(42)
device = "cuda"

FLASH_PREFILL_TARGETS = {
    4096: 74.0,
    8192: 53.0,
    16384: 32.2,
    32768: 17.8,
    65536: 9.0,
    131072: 4.5
}

SEQ_LENS = [4096, 8192, 16384, 32768, 65536, 131072]
SPLIT_VALS = [1]
CONCURRENCY_VALS = [4]

# Fixed sparse pattern parameters
ATTENTION_SINK = 2
WINDOW = 4
LAST_N_BLOCK = 0

# Selection granularity of the hand-built index (64 or 128). When larger than
# the kernel tile (64), the logical index is expanded into physical 64-token
# tiles so the epilogue correction's selected-block skipping stays aligned.
SEL_BLOCK_N = 128


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


def concurrent_fn(fn, streams):
    """Launch one independent request per CUDA stream and return all outputs."""
    outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            outputs.append(fn())
    return outputs


def benchmark_stage(name, fn, warmup, iters):
    try:
        return benchmark_fn(fn, warmup, iters)
    except Exception as exc:
        raise RuntimeError(f"{name} stage failed: {exc}") from exc


def run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
            seqlen_q, head_dim, softmax_scale, causal=True,
            block_sparse_cu=None, block_sparse_idx=None,
            total_q_tiles=None, cu_q_tiles=None,
            num_splits=0, sinks=None,
            k_mean=None, v_mean=None, mean_k_block_size=0):
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
    if sinks is not None:
        kwargs["sinks"] = sinks
    if k_mean is not None:
        kwargs.update(k_mean=k_mean, v_mean=v_mean,
                      mean_k_block_size=mean_k_block_size)
    return flash_attn_with_kvcache(**kwargs)


def build_block_means(k_cache, v_cache, page_table, cache_seqlens, kBlockN):
    """Pool per-block K/V means, layout (batch, max_k_blocks, nkv, headdim),
    matching what the Triton index builder produces for the correction."""
    batch = page_table.shape[0]
    page_size, nkv, d = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    max_k = int(cache_seqlens.max())
    nb = (max_k + kBlockN - 1) // kBlockN
    k_mean = torch.zeros(batch, nb, nkv, d, dtype=torch.float32, device=k_cache.device)
    v_mean = torch.zeros(batch, nb, nkv, d, dtype=torch.float32, device=k_cache.device)
    for b in range(batch):
        kv_len = int(cache_seqlens[b])
        npg = (kv_len + page_size - 1) // page_size
        pages = page_table[b, :npg].long()
        k_all = k_cache[pages].reshape(-1, nkv, d)[:kv_len].float()
        v_all = v_cache[pages].reshape(-1, nkv, d)[:kv_len].float()
        nbb = (kv_len + kBlockN - 1) // kBlockN
        pad = nbb * kBlockN - kv_len
        if pad:
            k_all = torch.cat([k_all, k_all.new_zeros(pad, nkv, d)])
            v_all = torch.cat([v_all, v_all.new_zeros(pad, nkv, d)])
        cnt = torch.ones(nbb * kBlockN, 1, 1, device=k_cache.device)
        if pad:
            cnt[kv_len:] = 0
        cnt = cnt.view(nbb, kBlockN).sum(1).clamp(min=1).view(nbb, 1, 1)
        k_mean[b, :nbb] = k_all.view(nbb, kBlockN, nkv, d).sum(1) / cnt
        v_mean[b, :nbb] = v_all.view(nbb, kBlockN, nkv, d).sum(1) / cnt
    return k_mean.to(k_cache.dtype), v_mean.to(v_cache.dtype)


def build_index_packgqa(
    batch_size, nheads_q, nheads_kv, seqlen_q, seqlen_k,
    kBlockM, kBlockN,
    attention_sink=ATTENTION_SINK, window=WINDOW, last_n_blocks=LAST_N_BLOCK,
    num_random_blocks=0, causal=True, device="cuda", rng_seed=42,
):
    """Build sparse index for PackGQA=True mode (kv head organized).
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
                        # Deterministic evenly-spaced selection instead of random
                        pick_indices = np.linspace(0, len(remaining) - 1, n_sel, dtype=int)
                        selected.update(remaining[i] for i in pick_indices)

                all_indices.extend(sorted(selected))
                cu_offsets.append(len(all_indices))

    block_sparse_cu = torch.tensor(cu_offsets, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(all_indices, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles


def expand_index_to_phys64(bs_cu, bs_idx, n_sub):
    """Expand a logical-block CSR (block size 64*n_sub) into physical
    64-token tile ids, mirroring the builder's N_SUB expansion so the
    epilogue correction's two-pointer scan stays aligned (a selected
    logical block always owns n_sub consecutive tiles)."""
    if n_sub == 1:
        return bs_cu, bs_idx
    counts = (bs_cu[1:] - bs_cu[:-1]).to(torch.int64)
    expanded_idx = (bs_idx.to(torch.int64)[:, None] * n_sub
                    + torch.arange(n_sub, device=bs_idx.device)).reshape(-1).to(torch.int32)
    expanded_cu = torch.zeros_like(bs_cu)
    torch.cumsum(counts * n_sub, dim=0, out=expanded_cu[1:])
    return expanded_cu, expanded_idx


def causal_tile_stats(bs_cu, bs_idx, batch, nheads_q, nheads_kv,
                      seqlen_q, seqlen_k, kBlockM, kBlockN, sel_n,
                      causal=True):
    """Return (selected, causal) physical kBlockN-tile counts.

    selected counts only causal-visible physical tiles owned by the selected
    logical blocks: a selected block straddling the causal boundary
    contributes just its visible tiles (clamp), so the ratio can never
    exceed 100%. causal is the total causal-visible tile count. Both use
    the first q position of each m_block, same convention as the kernel's
    per-tile iteration range.
    """
    device = bs_idx.device
    gqa_ratio = nheads_q // nheads_kv
    n_sub = max(1, sel_n // kBlockN)
    n_q_tiles = (seqlen_q * gqa_ratio + kBlockM - 1) // kBlockM
    n_k_tiles = (seqlen_k + kBlockN - 1) // kBlockN
    prefix_len = seqlen_k - seqlen_q

    q_pos = torch.arange(n_q_tiles, device=device) * kBlockM // gqa_ratio
    cap = (q_pos + prefix_len) // kBlockN + 1
    cap = cap.clamp(max=n_k_tiles) if causal else torch.full_like(cap, n_k_tiles)
    valid = q_pos < seqlen_q
    total_causal = int(cap[valid].sum()) * nheads_kv * batch

    # Segment order is (h_kv, b, m_block); empty segments have count 0.
    counts = (bs_cu[1:] - bs_cu[:-1]).to(torch.int64)
    seg_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device), counts)
    caps_seg = cap.to(torch.int64).repeat(nheads_kv * batch)
    contrib = (caps_seg[seg_ids] - bs_idx.to(torch.int64) * n_sub).clamp(0, n_sub)
    total_selected = int(contrib.sum())
    return total_selected, total_causal


def compute_sparsity(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                     kBlockM, kBlockN, sink=ATTENTION_SINK, window=WINDOW,
                     last_n=LAST_N_BLOCK, random_blocks=0, sel_n=None):
    sel_n = sel_n or kBlockN
    bs_cu, bs_idx, _, _ = build_index_packgqa(
        batch, nheads_q, nheads_kv, seqlen_q, seqlen_k, kBlockM, sel_n,
        attention_sink=sink, window=window, last_n_blocks=last_n,
        num_random_blocks=random_blocks, causal=True, device="cpu")
    total_selected, total_causal = causal_tile_stats(
        bs_cu, bs_idx, batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN, sel_n, causal=True)
    if total_causal == 0:
        return 0.0
    return total_selected / total_causal * 100


def find_best_config(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                     kBlockM, kBlockN, target_sparsity):
    """Search random_blocks closest to target sparsity, with fixed sink=1, window=2, last_n=2."""
    best = None
    best_diff = 999.0
    for rand in range(0, 33):
        sp = compute_sparsity(batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
                              kBlockM, kBlockN, ATTENTION_SINK, WINDOW, LAST_N_BLOCK, rand,
                              sel_n=SEL_BLOCK_N)
        diff = abs(sp - target_sparsity)
        if diff < best_diff:
            best_diff = diff
            best = (ATTENTION_SINK, WINDOW, LAST_N_BLOCK, rand, sp)
        if diff < 0.5:
            return best
    return best


def run_bench(dtype, head_dim, batch, seqlen_q, seqlen_k,
              nheads_q, nheads_kv, sink, window, last_n, random_blocks,
              num_splits=1, concurrency=1, warmup=3, iters=15, dense_cache=None):
    element_size = 1 if dtype == torch.float8_e4m3fn else 2
    kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)
    softmax_scale = head_dim ** (-0.5)
    dtype_name = "fp8" if dtype == torch.float8_e4m3fn else ("fp16" if dtype == torch.float16 else "bf16")

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = create_test_inputs(
        batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)

    sinks = torch.zeros(nheads_q, dtype=q.dtype, device=device)
    streams = [torch.cuda.Stream() for _ in range(concurrency)]

    # dense baseline: official cute flash_attn_varlen_func + FA2 (non-paged; dense k/v
    # gathered from the cache via page_table; only fp16/bf16 supported, so dense uses bf16 inputs for fp8 runs).
    # Input data is dtype-independent and reused across dtypes via dense_cache to avoid duplicate measurements.
    cache_key = (batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, concurrency)
    if dense_cache is not None and cache_key in dense_cache:
        med_dense, med_fa2 = dense_cache[cache_key]
    else:
        dtype_d = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
        q_d = q.to(dtype_d)
        k_d = k_cache[page_table.long()][:, :, 0].reshape(-1, nheads_kv, head_dim).to(dtype_d)
        v_d = v_cache[page_table.long()][:, :, 0].reshape(-1, nheads_kv, head_dim).to(dtype_d)
        cu_seqlens_k = torch.arange(0, (batch + 1) * seqlen_k, seqlen_k,
                                    dtype=torch.int32, device=device)
        fn_dense_single = lambda: fa_cute_varlen(
            q_d, k_d, v_d, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
            softmax_scale=softmax_scale, causal=True)
        med_dense = benchmark_stage(
            "dense attention (cute)", lambda: concurrent_fn(fn_dense_single, streams), warmup, iters
        )

        fn_fa2_single = lambda: fa2_varlen(
            q_d, k_d, v_d, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            softmax_scale=softmax_scale, causal=True)
        med_fa2 = benchmark_stage(
            "dense attention (fa2)", lambda: concurrent_fn(fn_fa2_single, streams), warmup, iters
        )
        if dense_cache is not None:
            dense_cache[cache_key] = (med_dense, med_fa2)

    sel_n = SEL_BLOCK_N
    n_sub = max(1, sel_n // kBlockN)
    bs_cu_log, bs_idx_log, total_q_tiles, cu_q_tiles = build_index_packgqa(
        batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, sel_n,
        attention_sink=sink, window=window, last_n_blocks=last_n,
        num_random_blocks=random_blocks, causal=True, device=device)
    bs_cu, bs_idx = expand_index_to_phys64(bs_cu_log, bs_idx_log, n_sub)

    flash_prefill_corr = FlashPrefill(
        k_block_m=kBlockM,
        k_block_n=sel_n,
        abs_threshold=1.0,
        attention_sink=sink,
        window_size=window,
        last_n_blocks=last_n,
        min_sparse_q_len=0,
        causal=True,
        softmax_scale=softmax_scale,
        num_splits=num_splits,
        use_mean_correction=True,
    )
    q_lens = (seqlen_q,) * batch

    def fn_index_select_corr_single():
        return flash_prefill_corr.index_select(
            q,
            k_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            v_cache=v_cache,
            q_lens=q_lens,
            max_cache_seqlen=seqlen_k,
            softmax_scale=softmax_scale,
        )
    med_index_select_corr = benchmark_stage(
        "index select (with V-mean pooling)", lambda: concurrent_fn(fn_index_select_corr_single, streams), warmup, iters
    )

    # Same hand-built index + pooled block means -> correction in the epilogue.
    k_mean, v_mean = build_block_means(k_cache, v_cache, page_table, cache_seqlens, sel_n)
    fn_sparse_corr_single = lambda: run_fa3(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                                            seqlen_q, head_dim, softmax_scale, causal=True,
                                            block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
                                            total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles,
                                            num_splits=num_splits, sinks=sinks,
                                            k_mean=k_mean, v_mean=v_mean,
                                            mean_k_block_size=sel_n)
    med_sparse_corr = benchmark_stage(
        "static sparse attention + mean correction", lambda: concurrent_fn(fn_sparse_corr_single, streams), warmup, iters
    )

    total_selected, total_causal = causal_tile_stats(
        bs_cu_log, bs_idx_log, batch, nheads_q, nheads_kv, seqlen_q, seqlen_k,
        kBlockM, kBlockN, sel_n, causal=True)
    sparsity_pct = total_selected / total_causal * 100 if total_causal > 0 else 0
    flops_ratio = total_causal / total_selected if total_selected > 0 else 1.0
    corr_speedup = med_dense / med_sparse_corr if med_sparse_corr > 0 else 0
    combined_ms = med_index_select_corr + med_sparse_corr
    combined_speedup = med_dense / combined_ms if combined_ms > 0 else 0
    fa2_combined_speedup = med_fa2 / combined_ms if combined_ms > 0 else 0
    pipeline_throughput = concurrency / combined_ms if combined_ms > 0 else 0

    return {
        "dtype": dtype_name, "hdim": head_dim, "batch": batch,
        "sq": seqlen_q, "num_splits": num_splits, "concurrency": concurrency,
        "dense_ms": med_dense * 1000,
        "fa2_ms": med_fa2 * 1000,
        "index_select_corr_ms": med_index_select_corr * 1000,
        "sparse_corr_ms": med_sparse_corr * 1000,
        "combined_ms": combined_ms * 1000,
        "sparsity_pct": sparsity_pct, "flops_ratio": flops_ratio,
        "corr_speedup": corr_speedup,
        "combined_speedup": combined_speedup,
        "fa2_combined_speedup": fa2_combined_speedup,
        "pipeline_throughput": pipeline_throughput,
    }


def main():
    NHEADS_Q = 32
    NHEADS_KV = 8  # GQA ratio = 8
    BATCH = 1

    for head_dim in [128]:
        dense_cache = {}  # dense/FA2 inputs are dtype-independent; reuse measurements across dtypes
        # for dtype, dtype_name in [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
        for dtype, dtype_name in [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]:
            element_size = 1 if dtype == torch.float8_e4m3fn else 2
            kBlockM, kBlockN = get_tile_sizes(head_dim, element_size, is_causal=True, paged_kv_non_tma=True)

            print(f"\n{'='*160}")
            print(f"  FlashPrefill Sparsity + last_n={LAST_N_BLOCK} Benchmark: {dtype_name}, hdim={head_dim}, GQA Q={NHEADS_Q} KV={NHEADS_KV} (ratio={NHEADS_Q//NHEADS_KV}), batch={BATCH}")
            print(f"  kBlockM={kBlockM}, kBlockN={kBlockN}, Forced num_splits: {SPLIT_VALS}")
            print(f"{'='*160}")

            configs = {}
            for sq in SEQ_LENS:
                target = FLASH_PREFILL_TARGETS[sq]
                best = find_best_config(BATCH, NHEADS_Q, NHEADS_KV, sq, sq, kBlockM, kBlockN, target)
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

                hdr = (f"  {'Conc':>4} {'Splits':>7} | {'Sparse%':>7} {'FLOPs_R':>7} | {'Dense ms':>9} {'Fa2 ms':>9} {'IdxSelC ms':>11} "
                       f"{'SparseC ms':>10} | {'CorrSpd':>8} {'CmbSpd':>8} {'Fa2Cmb':>8} {'req/s':>8}")
                print(hdr)
                print(f"  {'-'*125}")

                for concurrency in CONCURRENCY_VALS:
                    for ns in SPLIT_VALS:
                        try:
                            r = run_bench(dtype, head_dim, BATCH, sq, sq,
                                          NHEADS_Q, NHEADS_KV, sink, window, ln, rand,
                                          num_splits=ns, concurrency=concurrency,
                                          warmup=10, iters=10, dense_cache=dense_cache)
                            print(
                                f"  {concurrency:>4} sp={ns:>5} | {r['sparsity_pct']:>6.1f}% {r['flops_ratio']:>6.1f}x | "
                                f"{r['dense_ms']:>8.3f}m {r['fa2_ms']:>8.3f}m {r['index_select_corr_ms']:>10.3f}m "
                                f"{r['sparse_corr_ms']:>9.3f}m | "
                                f"{r['corr_speedup']:>7.2f}x {r['combined_speedup']:>7.2f}x {r['fa2_combined_speedup']:>7.2f}x {r['pipeline_throughput']:>7.1f}"
                            )
                        except Exception as e:
                            print(f"  {concurrency:>4} sp={ns:>5} | ERROR: {e}")
                            traceback.print_exc()
                        finally:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                print(f"  {'-'*125}")

    # print(f"\n{'='*160}")
    # print("Notes:")
    # print(f"  - Sparsity matched to FlashPrefill comparison table, last_n={LAST_N_BLOCK} (2 K-blocks always visible)")
    # print(f"  - GQA: Q={NHEADS_Q} KV={NHEADS_KV} (ratio={NHEADS_Q//NHEADS_KV}), causal=True, batch=1")
    # print("  - Block sparse uses packgqa index (kv head organized, PackGQA optimization enabled)")
    # print("  - Fixed: attention_sink=1, window=2, last_n_block=2")
    # print("  - Dense baseline always uses num_splits=1")
    # print(f"  - Concurrency levels: {CONCURRENCY_VALS}; one request per independent CUDA stream")
    # print("  - Concurrent requests share read-only Q/K/V inputs; all reported ms values are wall-clock batch latency")
    # print("  - FLOPs_R: theoretical FLOPs ratio (causal)")
    # print(f"  - Sparse%: selected / causal-visible {64}-token K-tiles; a selected {SEL_BLOCK_N}-token logical block")
    # print("    straddling the causal boundary counts only its visible tiles (ratio never exceeds 100%)")
    # print("  - IndexSel ms: FlashPrefill.index_select latency through the current public API")
    # print("  - IdxSelC ms: index_select with use_mean_correction=True (adds V block-mean pooling)")
    # print("  - Sparse ms: attention latency using the manually constructed target-density index")
    # print("  - SparseC ms: same index + pooled block means -> zero-order correction fused in the epilogue")
    # print("  - AttnSpd = concurrent Dense ms / concurrent Sparse ms; CorrSpd = Dense / SparseC; req/s = concurrency / Sparse seconds")
    # print(f"{'='*160}")

if __name__ == "__main__":
    main()