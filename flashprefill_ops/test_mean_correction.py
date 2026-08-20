# fp64-reference test for the zero-order mean correction of unselected blocks.
#
# The kernel must reproduce this semantics exactly:
#   * selected blocks (CSR) contribute exact per-token attention;
#   * unselected logical block j (fully visible to the whole 128-row packed
#     Q tile, i.e. (j+1)*k_block_n <= prefix + q_pos_min_of_tile) contributes
#     len_j * exp(q . k_mean_j * scale) to the denominator and
#     len_j * exp(q . k_mean_j * scale) * v_mean_j to the numerator;
#   * blocks intersecting the diagonal band are NOT corrected (they are
#     covered by the exact path through sink/local/last_n selection).

import torch

from flashprefill import FlashPrefill


def _gather_kv(k_cache, v_cache, page_table_b, kv_len, page_size):
    npages = (kv_len + page_size - 1) // page_size
    pages = page_table_b[:npages].long()
    k = k_cache[pages].reshape(-1, *k_cache.shape[2:])[:kv_len]
    v = v_cache[pages].reshape(-1, *v_cache.shape[2:])[:kv_len]
    return k, v


@torch.no_grad()
def reference_attention(
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
    block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles,
    k_mean, v_mean, page_size, k_block_m, k_block_n, scale, causal,
    correction: bool, full: bool,
):
    """fp64 reference. full=True ignores the index (dense exact attention)."""
    total_q, nheads, d = q.shape
    nkv = k_mean.shape[2] if k_mean is not None else k_cache.shape[2]
    gqa = nheads // nkv
    n_sub = k_block_n // 64
    dev = q.device
    out = torch.zeros(total_q, nheads, d, dtype=torch.float64, device=dev)
    batch = cache_seqlens.numel()
    for b in range(batch):
        q_lo, q_hi = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        qlen = q_hi - q_lo
        kv_len = int(cache_seqlens[b])
        prefix = kv_len - qlen
        k_all, v_all = _gather_kv(k_cache, v_cache, page_table[b], kv_len, page_size)
        k_all = k_all.double()
        v_all = v_all.double()
        n_logical = (kv_len + k_block_n - 1) // k_block_n
        if k_mean is not None:
            km = k_mean[b, :n_logical].double()
            vm = v_mean[b, :n_logical].double()
        lens = torch.clamp(
            kv_len - torch.arange(n_logical, device=dev) * k_block_n, max=k_block_n
        ).double()
        log_lens = torch.log(lens)
        n_tiles = (qlen * gqa + k_block_m - 1) // k_block_m
        for kvh in range(nkv):
            for tile in range(n_tiles):
                rows = torch.arange(
                    tile * k_block_m, min((tile + 1) * k_block_m, qlen * gqa), device=dev
                )
                q_pos = rows // gqa
                q_head = kvh * gqa + rows % gqa
                qr = q[q_lo + q_pos, q_head].double()  # (R, d)
                q_pos_min = tile * k_block_m // gqa

                if full:
                    toks = torch.arange(kv_len, device=dev)
                    unsel = None
                else:
                    seg = kvh * total_q_tiles + int(cu_q_tiles[b]) + tile
                    lo, hi = int(block_sparse_cu[seg]), int(block_sparse_cu[seg + 1])
                    sel_phys = block_sparse_idx[lo:hi].long()
                    if sel_phys.numel():
                        bounds = torch.stack(
                            [sel_phys * 64, torch.clamp(sel_phys * 64 + 64, max=kv_len)], dim=1
                        )
                        toks = torch.cat(
                            [torch.arange(a, b_, device=dev) for a, b_ in bounds.tolist()]
                        )
                    else:
                        toks = torch.empty(0, dtype=torch.long, device=dev)
                    sel_logical = torch.unique(sel_phys // n_sub)
                    if causal:
                        j_hi = min(n_logical, (prefix + q_pos_min) // k_block_n)
                    else:
                        j_hi = n_logical
                    unsel_mask = torch.ones(j_hi, dtype=torch.bool, device=dev)
                    unsel_mask[sel_logical[sel_logical < j_hi]] = False
                    unsel = torch.nonzero(unsel_mask, as_tuple=True)[0]

                # exact-token scores with per-token causal mask
                if toks.numel():
                    s_e = qr @ k_all[toks, kvh].T * scale  # (R, T)
                    if causal:
                        keep = toks[None, :] <= (prefix + q_pos)[:, None]
                        s_e = s_e.masked_fill(~keep, float("-inf"))
                else:
                    s_e = torch.full(
                        (rows.numel(), 0), float("-inf"), dtype=torch.float64, device=dev
                    )

                if correction and unsel is not None and unsel.numel():
                    s_u = qr @ km[unsel, kvh].T * scale + log_lens[unsel][None, :]
                else:
                    s_u = torch.full(
                        (rows.numel(), 0), float("-inf"), dtype=torch.float64, device=dev
                    )

                neg_inf = torch.full(
                    (rows.numel(), 1), float("-inf"), dtype=torch.float64, device=dev
                )
                m_e = s_e.max(dim=1, keepdim=True).values if s_e.shape[1] else neg_inf
                m_u = s_u.max(dim=1, keepdim=True).values if s_u.shape[1] else neg_inf
                m = torch.maximum(m_e, m_u)
                p_e = torch.exp(s_e - m)
                p_u = torch.exp(s_u - m)
                den = p_e.sum(dim=1) + p_u.sum(dim=1)
                num = torch.zeros(rows.numel(), d, dtype=torch.float64, device=dev)
                if toks.numel():
                    num += p_e @ v_all[toks, kvh]
                if s_u.shape[1]:
                    num += p_u @ vm[unsel, kvh]
                out[q_lo + q_pos, q_head] = num / den[:, None]
    return out


@torch.no_grad()
def run_case(dtype, k_block_n, causal=True, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    nheads, nkv, d = 8, 2, 128
    page_size = 16
    k_block_m = 128
    kv_lens = [8192, 6144]
    q_lens = [512, 300]
    batch = len(kv_lens)
    npages = sum((l + page_size - 1) // page_size for l in kv_lens)

    # fp8 tensors cannot be created with randn directly; generate in bf16
    # (and apply the hot-block scaling there) before casting.
    k_cache = torch.randn(npages, page_size, nkv, d, dtype=torch.bfloat16, device=dev)
    v_cache = torch.randn(npages, page_size, nkv, d, dtype=torch.bfloat16, device=dev)
    # Hot blocks: scale up K in ~30% of logical blocks so that attention is
    # concentrated and the sparse baseline carries non-trivial error (with
    # uniform random data the sparse baseline is already near-exact and the
    # correction's gain is invisible).
    hot_gen = torch.Generator().manual_seed(seed + 100)
    hot = torch.rand((max(kv_lens) + k_block_n - 1) // k_block_n, generator=hot_gen) < 0.3
    k_flat = k_cache.view(-1, nkv, d)
    row_ofs = 0
    for l in kv_lens:
        n_l = (l + k_block_n - 1) // k_block_n
        for j in range(n_l):
            if hot[j]:
                k_flat[row_ofs + j * k_block_n: row_ofs + min((j + 1) * k_block_n, l)] *= 3.0
        row_ofs += ((l + page_size - 1) // page_size) * page_size
    if dtype != torch.bfloat16:
        k_cache = k_cache.to(dtype)
        v_cache = v_cache.to(dtype)
    page_table = torch.zeros(batch, max((l + page_size - 1) // page_size for l in kv_lens),
                             dtype=torch.int32, device=dev)
    ofs = 0
    for b, l in enumerate(kv_lens):
        n = (l + page_size - 1) // page_size
        page_table[b, :n] = torch.arange(ofs, ofs + n, dtype=torch.int32, device=dev)
        ofs += n
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev
    )
    q = torch.randn(int(cu_seqlens_q[-1]), nheads, d, dtype=torch.bfloat16, device=dev).to(dtype)
    scale = d ** -0.5

    common = dict(
        k_block_m=k_block_m, k_block_n=k_block_n, causal=causal,
        softmax_scale=scale, abs_threshold=1.0, attention_sink=2, window_size=2,
        last_n_blocks=4, min_sparse_q_len=0,
    )
    attn_corr = FlashPrefill(use_mean_correction=True, **common)
    out_corr = attn_corr(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
        q_lens=q_lens, max_cache_seqlen=max(kv_lens),
    )
    attn_sparse = FlashPrefill(use_mean_correction=False, **common)
    out_sparse = attn_sparse(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
        q_lens=q_lens, max_cache_seqlen=max(kv_lens),
    )

    index = attn_corr.index_select(
        q, k_cache, page_table, cache_seqlens, cu_seqlens_q,
        v_cache=v_cache, q_lens=q_lens, max_cache_seqlen=max(kv_lens),
    )
    ref_corr = reference_attention(
        q.double(), k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
        index.block_sparse_cu, index.block_sparse_idx, index.total_q_tiles,
        index.cu_q_tiles, index.k_mean, index.v_mean,
        page_size, k_block_m, k_block_n, scale, causal,
        correction=True, full=False,
    )
    ref_full = reference_attention(
        q.double(), k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
        None, None, 0, index.cu_q_tiles, None, None,
        page_size, k_block_m, k_block_n, scale, causal,
        correction=False, full=True,
    )

    out_corr64 = out_corr.double()
    err_abs = (out_corr64 - ref_corr).abs()
    err_impl_mean = err_abs.mean().item()
    err_impl_max = err_abs.max().item()
    err_corr = (out_corr64 - ref_full).abs().mean().item()
    err_sparse = (out_sparse.double() - ref_full).abs().mean().item()
    print(
        f"dtype={str(dtype):25s} k_block_n={k_block_n} causal={causal} | "
        f"impl-vs-ref mean={err_impl_mean:.4e} max={err_impl_max:.4e} | mean|corr-full|={err_corr:.4e} "
        f"vs mean|sparse-full|={err_sparse:.4e} (improvement x{err_sparse / max(err_corr, 1e-12):.2f})"
    )
    # Mean error is the semantic-correctness metric; the max is dominated by
    # bf16 quantization noise in the hot-block regime (P/v_mean/output all
    # bf16, contributions amplified x3).
    tol_mean = 8e-3 if dtype == torch.bfloat16 else 2e-2
    tol_max = 0.15 if dtype == torch.bfloat16 else 0.3
    assert err_impl_mean < tol_mean, f"kernel does not match fp64 correction reference (mean): {err_impl_mean}"
    assert err_impl_max < tol_max, f"kernel does not match fp64 correction reference (max): {err_impl_max}"
    assert err_corr < err_sparse * 0.9, (
        f"correction did not meaningfully improve over plain sparse attention "
        f"({err_corr} vs {err_sparse})"
    )


@torch.no_grad()
def main():
    assert torch.cuda.is_available()
    run_case(torch.bfloat16, k_block_n=64)
    run_case(torch.bfloat16, k_block_n=128)
    run_case(torch.bfloat16, k_block_n=128, causal=True, seed=1)
    # fp8 e4m3 cache: same path, relaxed tolerance
    run_case(torch.float8_e4m3fn, k_block_n=128)
    print("OK")


if __name__ == "__main__":
    main()
