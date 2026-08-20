"""Smoke test for a flashprefill (FlashPrefill V2) installation.

Covers the main public APIs end to end on a paged KV cache:

  1. flash_attn_func            — dense contiguous attention
  2. flash_attn_with_kvcache    — dense paged KV cache attention
  3. FlashPrefill.index_select  — block-sparse index construction
  4. FlashPrefill.block_sparse_attention — sparse attention from a given index
  5. FlashPrefill.__call__      — full sparse pipeline (index + attention)
  6. __call__ with use_mean_correction=True — corrected sparse pipeline
  7. two-step high-level call (index_select + block_sparse_attention, corrected)
  8. low-level direct call (workspace + index kernel + flash_attn_with_kvcache)
  9. FP8 pipeline               — same pipeline on float8_e4m3fn tensors

All checks are smoke tests: the call must run and return finite output with
the expected shape and dtype. For numerical validation use the repo's
test_compare_fa3.py / test_mean_correction.py.

Usage:  python eval_install.py
"""

import sys

import torch

CHECKS = []
DEV = "cuda"
DTYPE = torch.bfloat16

# test shapes (Qwen3-30B-A3B-like GQA: 32 q heads, 4 kv heads, head_dim 128)
NQ, NKV, D, G = 32, 4, 128, 8
Q_LENS = [512, 256]
KV_LENS = [4096, 3763]
BLOCK_N = 128          # logical sparse block
BLOCK_M = 128          # query tile
SUB = BLOCK_N // 64    # physical 64-token tiles per logical block


def check(name, ok, info=""):
    CHECKS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


def smoke(name, out, shape, dtype=DTYPE):
    check(name,
          torch.isfinite(out.float()).all().item()
          and tuple(out.shape) == tuple(shape)
          and out.dtype == dtype,
          f"shape={tuple(out.shape)}, dtype={out.dtype}")


def make_batch(seed=0, dtype=DTYPE):
    """Random varlen bottom-right-causal batch on a contiguous paged KV cache."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    page = 1  # token-level paging (SGLang default); kernel has a dedicated page_size==1 path
    total_q = sum(Q_LENS)
    npages = sum((l + page - 1) // page for l in KV_LENS)
    q = torch.randn(total_q, NQ, D, generator=g, device=DEV, dtype=dtype)
    k_cache = torch.randn(npages, page, NKV, D, generator=g, device=DEV, dtype=dtype)
    v_cache = torch.randn(npages, page, NKV, D, generator=g, device=DEV, dtype=dtype)
    max_pages = max((l + page - 1) // page for l in KV_LENS)
    page_table = torch.zeros(len(Q_LENS), max_pages, dtype=torch.int32, device=DEV)
    ofs = 0
    for b in range(len(Q_LENS)):
        n = (KV_LENS[b] + page - 1) // page
        page_table[b, :n] = torch.arange(ofs, ofs + n, dtype=torch.int32, device=DEV)
        ofs += n
    cache_seqlens = torch.tensor(KV_LENS, dtype=torch.int32, device=DEV)
    cum = torch.cumsum(torch.tensor([0] + Q_LENS), 0)
    cu_seqlens_q = cum.to(torch.int32).to(DEV)
    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(1)

    # ---- 0. import & compiled extension ----
    import flashprefill
    from flashprefill import FlashPrefill
    from flashprefill.flash_attn_interface import (
        flash_attn_func, flash_attn_with_kvcache)
    check("import flashprefill", True, f"version={flashprefill.__version__}")
    import flashprefill._C  # noqa
    check("compiled extension flashprefill._C", True)

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_batch()

    # ---- 1. flash_attn_func: dense contiguous ----
    qd = torch.randn(2, 1024, NQ, D, device=DEV, dtype=DTYPE)
    kd = torch.randn(2, 1024, NKV, D, device=DEV, dtype=DTYPE)
    vd = torch.randn(2, 1024, NKV, D, device=DEV, dtype=DTYPE)
    out = flash_attn_func(qd, kd, vd, causal=True)
    smoke("flash_attn_func (dense contiguous, causal)", out, (2, 1024, NQ, D))

    # ---- 2. flash_attn_with_kvcache: dense paged ----
    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max(Q_LENS), causal=True)
    smoke("flash_attn_with_kvcache (dense paged, varlen)", out, (sum(Q_LENS), NQ, D))

    # ---- 3. index_select ----
    fp = FlashPrefill(k_block_m=BLOCK_M, k_block_n=BLOCK_N, abs_threshold=0.1,
                      attention_sink=2, window_size=4, last_n_blocks=8)
    index = fp.index_select(q, k_cache, page_table, cache_seqlens, cu_seqlens_q,
                            q_lens=Q_LENS)
    cu = index.block_sparse_cu.cpu()
    n_seg = len(cu) - 1
    density = (cu[-1].double() / n_seg / (max(KV_LENS) / 64)).item()
    check("FlashPrefill.index_select (CSR index)",
          index.block_sparse_idx.dtype == torch.int32 and cu[-1].item() > 0
          and 0.0 < density <= 1.0,
          f"segments={n_seg}, density~{density:.3f}")

    # ---- 4. block_sparse_attention with an explicit index ----
    out = fp.block_sparse_attention(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, index,
        max_seqlen_q=max(Q_LENS))
    smoke("FlashPrefill.block_sparse_attention", out, (sum(Q_LENS), NQ, D))

    # ---- 5. __call__ full sparse pipeline ----
    out = fp(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
             q_lens=Q_LENS)
    smoke("FlashPrefill.__call__ (sparse pipeline, no correction)",
          out, (sum(Q_LENS), NQ, D))

    # ---- 6. __call__ with mean correction ----
    fpc = FlashPrefill(k_block_m=BLOCK_M, k_block_n=BLOCK_N, abs_threshold=0.1,
                       attention_sink=2, window_size=4, last_n_blocks=8,
                       use_mean_correction=True)
    out_c = fpc(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                q_lens=Q_LENS)
    smoke("FlashPrefill.__call__ with use_mean_correction=True",
          out_c, (sum(Q_LENS), NQ, D))

    # ---- 7. two-step high-level call (index_select + block_sparse_attention) ----
    idx_c = fpc.index_select(q, k_cache, page_table, cache_seqlens, cu_seqlens_q,
                             v_cache=v_cache, q_lens=Q_LENS)
    out_c2 = fpc.block_sparse_attention(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, idx_c,
        max_seqlen_q=max(Q_LENS))
    smoke("two-step call: index_select + block_sparse_attention (corrected)",
          out_c2, (sum(Q_LENS), NQ, D))

    # ---- 8. low-level direct call ----
    from flash_block_sparse_index_triton import (
        SparseIndexWorkspace, build_block_sparse_index_fast)
    q_tiles = [(l * G + BLOCK_M - 1) // BLOCK_M for l in Q_LENS]
    cu_q_tiles = torch.tensor([0, *torch.tensor(q_tiles).cumsum(0).tolist()],
                              dtype=torch.int32, device=DEV)
    ws = SparseIndexWorkspace(
        batch_size=2, num_kv_heads=NKV, head_dim=D,
        total_q_tiles=sum(q_tiles), max_q_tiles=max(q_tiles),
        max_k_blocks=(max(KV_LENS) + BLOCK_N - 1) // BLOCK_N,
        dtype=DTYPE, device=k_cache.device,
        cu_q_tiles=cu_q_tiles, n_sub=SUB, use_mean_correction=True)
    cu_l, idx_l, total_tiles_l, cu_q_tiles_l = build_block_sparse_index_fast(
        q, k_cache, page_table, cache_seqlens, cu_seqlens_q, ws,
        v_cache=v_cache, k_block_m=BLOCK_M, k_block_n=BLOCK_N,
        abs_threshold=0.1, attention_sink=2, window_size=4, last_n_blocks=8,
        causal=True)
    out_l = flash_attn_with_kvcache(
        q, k_cache, v_cache, page_table=page_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max(Q_LENS), causal=True,
        block_sparse_cu=cu_l, block_sparse_idx=idx_l,
        total_q_tiles=total_tiles_l, cu_q_tiles=cu_q_tiles_l,
        k_mean=ws.k_mean, v_mean=ws.v_mean, mean_k_block_size=BLOCK_N)
    smoke("low-level direct call (workspace + index kernel + flash_attn_with_kvcache)",
          out_l, (sum(Q_LENS), NQ, D))

    # ---- 9. FP8 pipeline ----
    out8 = fp(q.to(torch.float8_e4m3fn), k_cache.to(torch.float8_e4m3fn),
              v_cache.to(torch.float8_e4m3fn), page_table, cache_seqlens,
              cu_seqlens_q, q_lens=Q_LENS)
    smoke("FP8 pipeline (float8_e4m3fn)", out8, (sum(Q_LENS), NQ, D))

    print()
    n_fail = CHECKS.count(False)
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed"
          + (" — installation OK" if n_fail == 0 else " — FAILURES PRESENT"))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
