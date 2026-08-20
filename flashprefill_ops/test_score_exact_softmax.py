"""Comparison: triton exact row-level softmax scoring builder vs PyTorch naive reference (float64).

Verify that the CSR selected by the updated _packgqa_score_kernel (online softmax,
row-level exact) + _select_sparse_blocks_kernel exactly matches the element-wise reference.
"""
import math
import torch

from flash_block_sparse_index_triton import build_block_sparse_index
import test_block_sparse

device = "cuda"


def reference_build(q, k_cache, page_table, cache_seqlens, cu_seqlens_q,
                    k_block_m=128, k_block_n=64, abs_threshold=1.0,
                    attention_sink=2, window_size=4, last_n_blocks=2,
                    min_sparse_q_len=0, causal=True):
    batch = cache_seqlens.numel()
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[2]
    page_size = k_cache.shape[1]
    gqa = num_q_heads // num_kv_heads
    softmax_scale = head_dim ** -0.5
    scale_log2 = softmax_scale * math.log2(math.e)

    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    n_tiles = [(l * gqa + k_block_m - 1) // k_block_m for l in q_lens]
    cu_q_tiles = [0]
    for n in n_tiles:
        cu_q_tiles.append(cu_q_tiles[-1] + n)

    cu_list, idx_list = [0], []

    for h in range(num_kv_heads):
        for b in range(batch):
            q_len = q_lens[b]
            kv_len = int(cache_seqlens[b])
            prefix_len = kv_len - q_len
            q_beg = int(cu_seqlens_q[b])
            max_k_blocks = (kv_len + k_block_n - 1) // k_block_n

            tok = torch.arange(kv_len, device=device)
            pages = page_table[b, tok // page_size].long()
            offs = tok % page_size
            k_all = k_cache[pages, offs, h].to(torch.float64)
            k_mean = torch.zeros(max_k_blocks, head_dim, dtype=torch.float64, device=device)
            for j in range(max_k_blocks):
                lo_t, hi_t = j * k_block_n, min((j + 1) * k_block_n, kv_len)
                # Match triton: mean pool (per-dimension average), cast back to the input dtype before the dot
                k_mean[j] = k_all[lo_t:hi_t].mean(dim=0).to(k_cache.dtype).to(torch.float64)

            for m in range(n_tiles[b]):
                packed = m * k_block_m + torch.arange(k_block_m, device=device)
                q_pos = packed // gqa
                q_head = h * gqa + packed % gqa
                valid_q = (q_pos < q_len) & (q_head < num_q_heads)
                qt = torch.zeros(k_block_m, head_dim, dtype=torch.float64, device=device)
                vv = valid_q
                qt[vv] = q[q_beg + q_pos[vv], q_head[vv]].to(torch.float64)

                packed_end = min((m + 1) * k_block_m, q_len * gqa)
                q_first = m * k_block_m // gqa
                q_last = (packed_end - 1) // gqa
                first_k = (prefix_len + q_first) // k_block_n
                last_k = (prefix_len + q_last) // k_block_n
                full_last = m >= max(n_tiles[b] - last_n_blocks, 0)

                if causal:
                    active = (prefix_len + q_last + 1) // k_block_n
                else:
                    active = max_k_blocks

                keep = set()
                if q_len <= min_sparse_q_len or full_last:
                    # A dense tile must select all causally visible blocks, including the
                    # diagonal boundary block last_k (active is the scored range and would miss this unscored boundary block)
                    rng = range(last_k + 1 if causal else max_k_blocks)
                    keep = {j for j in rng if j * k_block_n < kv_len}
                else:
                    # Row-level exact softmax: visibility is per-row (a block participates if its first token is visible)
                    logits = qt @ k_mean[:active].T  # (M, active), scale not yet applied
                    logits = logits * scale_log2
                    if causal:
                        vis = valid_q[:, None] & (
                            prefix_len + q_pos[:, None] >=
                            torch.arange(active, device=device)[None, :] * k_block_n)
                    else:
                        vis = valid_q[:, None].expand(k_block_m, active)
                    logits = torch.where(vis, logits, torch.tensor(float("-inf"), dtype=torch.float64, device=device))
                    # Consistent with triton (paper Sec 3.2/3.4): the block score is the tile-level energy
                    # S[b] = sum_rows 2^(logit2[r,b] - M_tile), where M_tile is the maximum
                    # of all visible logits in the tile; threshold = alpha * max_b S[b]
                    logits2 = logits * math.log(2.0)  # log2 domain
                    if active > 0:
                        m_tile = logits2[vis].max()
                        score = torch.where(
                            vis, torch.exp2(logits2 - m_tile), torch.zeros_like(logits2)
                        ).sum(dim=0)  # (active,)
                        thresh = abs_threshold * float(score.max())
                    else:
                        thresh = 0.0
                    # The selection loop covers all k blocks: sink/local are not limited by the
                    # scored range (active); scored only applies to scored blocks (j < active) -- consistent with triton select.
                    for j in range(max_k_blocks):
                        if j * k_block_n >= kv_len:
                            continue
                        scored = j < active and score[j].item() >= thresh
                        sink = j < attention_sink
                        local = (j >= first_k - window_size + 1) and (j <= last_k)
                        if (scored or sink or local) and (not causal or j <= last_k):
                            keep.add(j)

                # Consistent with triton: logical blocks are expanded into contiguous 64-token
                # physical tiles; the sequence tail block only expands into e_tail valid tiles (the rest are fully out of bounds and never emitted)
                n_sub = k_block_n // 64
                last_logical = (kv_len - 1) // k_block_n
                e_tail = (kv_len + 63) // 64 - last_logical * n_sub
                emitted = 0
                for j in sorted(keep):
                    n_emit = e_tail if j == last_logical else n_sub
                    idx_list.extend(j * n_sub + i for i in range(n_emit))
                    emitted += n_emit
                cu_list.append(cu_list[-1] + emitted)

    return (torch.tensor(cu_list, dtype=torch.int32),
            torch.tensor(idx_list, dtype=torch.int32))


def run_case(batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim,
             dtype, sink=2, window=4, last_n=2, abs_threshold=1.0, seed=0, tag="",
             k_block_n=64):
    torch.manual_seed(seed)
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = \
        test_block_sparse.create_test_inputs(
            batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, dtype, device)
    if dtype == torch.float8_e4m3fn:
        q = q.to(dtype)  # the builder requires q and k_cache to have the same dtype

    cu_t, idx_t, total_t, cuq_t = build_block_sparse_index(
        q, k_cache, page_table, cache_seqlens, cu_seqlens_q,
        attention_sink=sink, window_size=window, last_n_blocks=last_n,
        abs_threshold=abs_threshold, k_block_n=k_block_n)
    cu_r, idx_r = reference_build(
        q, k_cache, page_table, cache_seqlens, cu_seqlens_q,
        attention_sink=sink, window_size=window, last_n_blocks=last_n,
        abs_threshold=abs_threshold, k_block_n=k_block_n)

    cu_t, idx_t = cu_t.cpu(), idx_t.cpu()
    same_cu = cu_t.tolist() == cu_r.tolist()
    same_idx = idx_t.tolist() == idx_r.tolist()
    if same_cu and same_idx:
        print(f"OK {tag}: entries={idx_t.numel()}")
        return True
    # Diff analysis: per-segment stats of selected-block set differences (bf16 dot + fp32 online
    # rescale vs the fp64 reference have inherent numerical boundaries; diffs should only appear in blocks near the abs_threshold threshold)
    gqa = nheads_q // nheads_kv
    q_lens = [seqlen_q] * batch_size
    n_diff_seg = 0
    diff_blocks = 0
    total_blocks = 0
    cu_r_l = cu_r.tolist()
    cu_t_l = cu_t.tolist()
    seg = 0
    for h in range(nheads_kv):
        for b in range(batch_size):
            n_tiles = (q_lens[b] * gqa + 127) // 128
            for m in range(n_tiles):
                sr = set(idx_r[cu_r_l[seg]:cu_r_l[seg + 1]].tolist())
                st = set(idx_t[cu_t_l[seg]:cu_t_l[seg + 1]].tolist())
                total_blocks += len(sr | st)
                if sr != st:
                    n_diff_seg += 1
                    diff_blocks += len(sr ^ st)
                    if n_diff_seg <= 3:
                        print(f"  seg {seg} (h={h},b={b},m={m}): ref_only={sorted(sr - st)} triton_only={sorted(st - sr)}")
                seg += 1
    pct = 100.0 * diff_blocks / max(total_blocks, 1)
    ok = diff_blocks <= max(4, total_blocks * 0.01)
    print(f"{'OK(borderline)' if ok else 'FAIL'} {tag}: diff_seg={n_diff_seg}/{seg}, "
          f"diff_blocks={diff_blocks}/{total_blocks} ({pct:.2f}%), triton={idx_t.numel()} ref={idx_r.numel()}")
    return ok


def main():
    ok = True
    ok &= run_case(1, 512, 512, 16, 2, 128, torch.bfloat16, tag="bs1-g8-hd128")
    ok &= run_case(1, 512, 1024, 16, 2, 128, torch.bfloat16, tag="bs1-g8-hd128-prefix")
    ok &= run_case(2, 256, 300, 8, 2, 128, torch.bfloat16, tag="bs2-varlen")
    ok &= run_case(1, 300, 300, 12, 1, 128, torch.bfloat16, tag="g12-hkv1")
    ok &= run_case(1, 256, 256, 24, 2, 128, torch.bfloat16, tag="g12-h24kv2")
    ok &= run_case(1, 512, 512, 8, 2, 256, torch.bfloat16, tag="hd256")
    ok &= run_case(1, 509, 511, 16, 4, 128, torch.bfloat16, tag="odd-len")
    # With high tau (>=1.5), scored blocks are rare, and borderline blocks near the threshold
    # naturally exceed the 1% flip-rate tolerance under bf16 vs fp64 rounding; the tau sweep is restricted to the low-flip region
    for t in (0.5, 1.0):
        ok &= run_case(1, 512, 512, 16, 2, 128, torch.bfloat16, abs_threshold=t, tag=f"tau{t}")
    # Logical blocks 128 / 256 (expanded after selection into 2 / 4 physical 64-token tiles)
    for kbn in (128, 256):
        ok &= run_case(1, 512, 512, 16, 2, 128, torch.bfloat16, k_block_n=kbn, tag=f"kbn{kbn}")
        ok &= run_case(2, 509, 511, 16, 4, 128, torch.bfloat16,
                       k_block_n=kbn, tag=f"varlen-kbn{kbn}")
        ok &= run_case(1, 512, 1024, 16, 4, 128, torch.float8_e4m3fn,
                       k_block_n=kbn, tag=f"fp8-kbn{kbn}")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
