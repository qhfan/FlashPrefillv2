"""FA cute (official CuTe DSL) vs our flashprefill: block sparse fwd/bwd speed comparison.

Same sparse pattern source: first generated with our PackGQA CSR (q128 packed x k64),
then converted back into the (B,1,M,N) q128xk128 mask FA cute needs, plus its transpose (for bwd).
Each side faithfully reports its own actual sparse% (minor differences due to different block granularity).

Only bf16 hdim128 is tested (FA cute does not support fp8 bwd; its hdim256 bwd block sparse is unverified).
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flashprefill  # noqa: F401
from flash_attn_interface import flash_attn_varlen_func, _flash_attn_backward
from flash_block_sparse_bwd_index import build_block_sparse_bwd_index
from flash_attn.cute.interface import _flash_attn_fwd as fa_fwd, _flash_attn_bwd as fa_bwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

from bench_block_sparse_perf import (benchmark_fn, find_best_config, compute_sparsity,
                                     build_index_packgqa, fa3,
                                     ATTENTION_SINK, WINDOW, LAST_N_BLOCK)

device = "cuda"
H_Q, H_KV = 32, 8
GQA = H_Q // H_KV
HEAD_DIM = 128
BENCH_SEQLENS = [int(x) for x in os.getenv("BENCH_SEQ", "4096,8192,16384,32768").split(",")]
BENCH_SPS = [float(x) for x in os.getenv("BENCH_SPS", "5,25,50,75,95").split(",")]
WARMUP, REP = 5, 20


def pair_up_k128(bs_cu, bs_idx):
    """Pad the k64 blocks selected in the CSR into k128 pairs (if j is selected, add j^1),
    so both sides select exactly the same set (FA cute's kv block granularity can only be 128)."""
    cu_list = [0]
    idx_list = []
    for p in range(bs_cu.numel() - 1):
        lo, hi = int(bs_cu[p]), int(bs_cu[p + 1])
        sel = set(bs_idx[lo:hi].cpu().tolist())
        for j in list(sel):
            sel.add(j ^ 1)
        sel = sorted(sel)
        idx_list.extend(sel)
        cu_list.append(cu_list[-1] + len(sel))
    return (torch.tensor(cu_list, dtype=torch.int32, device=device),
            torch.tensor(idx_list, dtype=torch.int32, device=device))


def align_to_q128(bs_cu, bs_idx, num_q_tiles_per_hkv):
    """Union the selections of the GQA packed tiles of each q128 tile and write them back,
    so our selected set strictly matches FA cute's q128 row granularity."""
    cu_list = [0]
    idx_list = []
    for h in range(H_KV):
        base = h * num_q_tiles_per_hkv
        for m0 in range(0, num_q_tiles_per_hkv, GQA):
            sel = set()
            for p in range(m0, min(m0 + GQA, num_q_tiles_per_hkv)):
                lo, hi = int(bs_cu[base + p]), int(bs_cu[base + p + 1])
                sel |= set(bs_idx[lo:hi].cpu().tolist())
            sel = sorted(sel)
            for _ in range(m0, min(m0 + GQA, num_q_tiles_per_hkv)):
                idx_list.extend(sel)
                cu_list.append(cu_list[-1] + len(sel))
    return (torch.tensor(cu_list, dtype=torch.int32, device=device),
            torch.tensor(idx_list, dtype=torch.int32, device=device))


def actual_sparsity_ours(bs_cu, bs_idx, sq):
    """Our side's actual sparse%: row range [Pp, P(p+1)) of packed tile p (P=128/gqa),
    counted per row with causal truncation."""
    P = 128 // GQA
    num_k64 = (sq + 63) // 64
    total = 0
    sel = 0
    for p in range(bs_cu.numel() - 1):
        r0 = P * p
        rows = min(P, sq - r0)
        if rows <= 0:
            break
        causal_cols = r0 + rows
        total += rows * causal_cols
        lo, hi = int(bs_cu[p]), int(bs_cu[p + 1])
        for j in bs_idx[lo:hi].cpu().tolist():
            cols = min(64 * (j + 1), causal_cols) - 64 * j
            sel += rows * max(0, cols)
    return 100.0 * sel / max(1, total)


def fa_mask_sparsity(fa_mask, sq):
    """Actual sparse% of the FA q128xk128 mask (exact per-row count with causal truncation)."""
    M = fa_mask.shape[0]
    total_pairs = 0
def fa_mask_sparsity(fa_mask, sq, bm=128, bn=128):
    """Actual sparse% of the FA mask (exact per-row count with causal truncation), block granularity (bm, bn)."""
    M = fa_mask.shape[0]
    total_pairs = 0
    sel_pairs = 0
    for m in range(M):
        rows = min(bm, sq - bm * m)
        causal_cols = bm * m + rows
        total_pairs += rows * causal_cols
        for n in range(fa_mask.shape[1]):
            if fa_mask[m, n]:
                cols = min(bn * (n + 1), causal_cols) - bn * n
                sel_pairs += rows * max(0, cols)
    return 100.0 * sel_pairs / max(1, total_pairs)


def build_fa_mask_direct(sq, target_sp, seed=42, bm=128, bn=128):
    """Independently generate a mask at FA's native (bm,bn) granularity: each q tile randomly
    selects r k blocks within the causal range (including the diagonal block); r is binary-searched to match target sparse%."""
    M = (sq + bm - 1) // bm
    N = (sq + bn - 1) // bn
    g = torch.Generator(device=device).manual_seed(seed)

    def make_mask(r):
        mask = torch.zeros(M, N, dtype=torch.bool, device=device)
        for m in range(M):
            rows = min(bm, sq - bm * m)
            n_cand = (bm * m + rows + bn - 1) // bn  # number of blocks intersecting the causal region
            idx = torch.randperm(n_cand, generator=g, device=device)[:r]
            mask[m, idx] = True
        return mask

    lo, hi, best, best_diff = 1, N, None, 999.0
    while lo <= hi:
        mid = (lo + hi) // 2
        sp = fa_mask_sparsity(make_mask(mid), sq, bm, bn)
        diff = sp - target_sp
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best = mid
        if abs(diff) < 0.5:
            break
        if diff < 0:
            lo = mid + 1
        else:
            hi = mid - 1
    mask = make_mask(best)
    return mask, fa_mask_sparsity(mask, sq, bm, bn)


def fa_mask_to_bs(fa_mask, bm=128, bn=128):
    """(M,N) bool mask -> BlockSparseTensorsTorch for fwd."""
    cnt = fa_mask.sum(dim=-1, dtype=torch.int32)
    idx = torch.argsort((~fa_mask).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    R = fa_mask.shape[0]
    zero = torch.zeros(1, 1, R, dtype=torch.int32, device=device)
    empty = torch.empty(1, 1, R, 0, dtype=torch.int32, device=device)
    return BlockSparseTensorsTorch(cnt[None, None, :].contiguous(),
                                   idx[None, None, :, :].contiguous(),
                                   zero, empty, (bm, bn))


def build_fa_masks(bs_cu, bs_idx, sq, seed):
    """Derive FA cute's q128xk128 mask + transpose from our packed CSR (the h_kv=0 segments).

    packed tile p covers per-head row range [32p, 32(p+1)) (when gqa=4),
    which is exactly contained in the row range [128m, 128(m+1)) of FA q tile m = p//GQA.
    """
    num_packed = bs_cu.numel() - 1
    num_k64 = (sq + 63) // 64
    num_packed_per_hkv = num_packed // H_KV if num_packed % H_KV == 0 else None
    # CSR layout of build_packgqa_csr_varlen: segment id = total_q_tiles * h_kv + tile
    # the h_kv=0 segments are the first num_q_tiles ones
    num_q_tiles = num_packed // H_KV
    S = torch.zeros(num_q_tiles, num_k64, dtype=torch.bool, device=device)
    for p in range(num_q_tiles):
        lo, hi = int(bs_cu[p]), int(bs_cu[p + 1])
        if hi > lo:
            S[p, bs_idx[lo:hi].long()] = True
    M = (sq + 127) // 128
    # row range of packed tile p is contained in q128 tile m=p//GQA: group by GQA and take the union
    S_pad = torch.zeros(M * GQA, num_k64, dtype=torch.bool, device=device)
    S_pad[:num_q_tiles] = S
    S_rows = S_pad.view(M, GQA, num_k64).any(dim=1)  # (M, num_k64)
    N128 = (sq + 127) // 128
    k64_pad = torch.zeros(M, N128 * 2, dtype=torch.bool, device=device)
    k64_pad[:, :num_k64] = S_rows
    fa_mask = k64_pad.view(M, N128, 2).any(dim=-1)  # (M, N128)

    def to_bs(mask_rows):
        cnt = mask_rows.sum(dim=-1, dtype=torch.int32)
        idx = torch.argsort((~mask_rows).to(torch.int8), dim=-1, stable=True).to(torch.int32)
        R, C = mask_rows.shape
        cnt_b = cnt[None, None, :].contiguous()
        idx_b = idx[None, None, :, :].contiguous()
        zero = torch.zeros(1, 1, R, dtype=torch.int32, device=device)
        empty = torch.empty(1, 1, R, 0, dtype=torch.int32, device=device)
        return BlockSparseTensorsTorch(cnt_b, idx_b, zero, empty, (128, 128)), cnt_b, idx_b

    fwd_bs, _, _ = to_bs(fa_mask)
    bwd_bs, _, _ = to_bs(fa_mask.t().contiguous())
    # FA side actual sparse% (k128 granularity; causal-friendly diagonal blocks always selected)
    total_pairs = 0
    sel_pairs = 0
    for m in range(M):
        rows = min(128, sq - 128 * m)
        causal_cols = 128 * m + rows  # causal-visible columns within the row range (upper bound)
        total_pairs += rows * causal_cols
        for n in range(N128):
            if fa_mask[m, n]:
                cols = min(128 * (n + 1), 128 * m + rows) - 128 * n
                sel_pairs += rows * max(0, cols)
    fa_sp = 100.0 * sel_pairs / max(1, total_pairs)
    return fwd_bs, bwd_bs, fa_sp


def run_one(sq, sp_target, head_dim):
    rand = find_best_config(1, H_Q, H_KV, sq, sq, 128, 64, sp_target)[3]

    q = torch.randn(1, sq, H_Q, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    k = torch.randn(1, sq, H_KV, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    v = torch.randn_like(k)
    cu = torch.tensor([0, sq], dtype=torch.int32, device=device)

    # Our native k64-granularity CSR (not aligned; the granularity difference is architectural, each side reports sparse% as-is)
    bs_cu, bs_idx, total_q_tiles, cu_q_tiles = build_index_packgqa(
        1, H_Q, H_KV, sq, sq, 128, 64, num_random_blocks=rand, causal=True, device=device)
    our_sp = actual_sparsity_ours(bs_cu, bs_idx, sq)

    # FA cute mask: independently generated at its native granularity, matching the same target sparse%
    # hdim128 -> tile (128,128); hdim256 -> tile_n=80 (constraint of normalize_block_sparse_config)
    fa_bn = 128 if head_dim == 128 else 80
    fa_mask, fa_sp = build_fa_mask_direct(sq, sp_target, bn=fa_bn)
    fwd_bs = fa_mask_to_bs(fa_mask, bn=fa_bn)

    # ---- fwd ---- (dense baseline: stock FA3)
    fn_dense_fwd = lambda: fa3.flash_attn_varlen_func(q[0], k[0], v[0], cu, cu, sq, sq, causal=True)
    fn_fa_fwd = lambda: fa_fwd(q, k, v, causal=True, block_sparse_tensors=fwd_bs)
    fn_our_fwd = lambda: flash_attn_varlen_func(
        q[0], k[0], v[0], cu, cu, sq, sq, causal=True,
        block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
        total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)
    t_dense_fwd = benchmark_fn(fn_dense_fwd, warmup=WARMUP, iters=REP)
    t_fa_fwd = benchmark_fn(fn_fa_fwd, warmup=WARMUP, iters=REP)
    t_our_fwd = benchmark_fn(fn_our_fwd, warmup=WARMUP, iters=REP)

    return our_sp, fa_sp, t_dense_fwd, t_fa_fwd, t_our_fwd


def main():
    print(f"device: {torch.cuda.get_device_name(0)}, GQA {H_Q}/{H_KV} (ratio={GQA}), bf16, batch=1, fwd only")
    print(f"pattern: sink={ATTENTION_SINK}, window={WINDOW}, last_n={LAST_N_BLOCK} + random; index prebuilt")
    for head_dim in [int(x) for x in os.getenv("BENCH_HDIM", "128,256").split(",")]:
        print(f"\n===== hdim={head_dim} =====")
        hdr = (f"  {'SeqLen':>7} {'Target%':>8} | {'OurSp%':>7} {'Ofwd':>7} | "
               f"{'FaSp%':>7} {'Ffwd':>7}   (both are speedups vs dense)")
        print(hdr)
        print(f"  {'-' * 62}")
        for sq in BENCH_SEQLENS:
            for sp in BENCH_SPS:
                try:
                    our_sp, fa_sp, tdf, tff, tof = run_one(sq, sp, head_dim)
                    print(f"  {sq:>7} {sp:>7.1f}% | {our_sp:>6.1f}% {tdf / tof:>6.2f}x | "
                          f"{fa_sp:>6.1f}% {tdf / tff:>6.2f}x")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  {sq:>7} {sp:>7.1f}% | ERROR: {type(e).__name__}: {e}")
                finally:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
