"""
Check the numerical correctness of flash_attn_varlen_func block sparse backward.

Method:
- Build a PackGQA CSR with sink+window (k_block_m=128, k_block_n=64),
  consistent with the semantics of the forward kernel.
- kernel: flash_attn_varlen_func(..., causal=True, block_sparse_*) backward directly.
- reference: fp32 masked attention, mask = (k blocks selected by the CSR) AND (bottom-right causal),
  with dq/dk/dv computed by autograd.
- Compare rel L2: ||a-b||/||b||.

Also tested: K positions not selected by any segment must have dk/dv strictly equal to 0.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_block_sparse import m_block_to_q_pos, q_pos_to_k_block  # noqa: E402
from flash_attn_interface import flash_attn_varlen_func  # noqa: E402

K_BLOCK_M = 128
K_BLOCK_N = 64
HEAD_DIM = 128


def build_packgqa_csr_varlen(seqlens_q, seqlens_k, nheads_q, nheads_kv,
                             sink=1, window=2, device="cuda"):
    """PackGQA CSR，segment = h_kv * total_q_tiles + (cu_q_tiles[b] + packed_m128)。"""
    gqa = nheads_q // nheads_kv
    batch = len(seqlens_q)
    n_q_tiles = [(sq * gqa + K_BLOCK_M - 1) // K_BLOCK_M for sq in seqlens_q]
    cu_q_tiles = [0]
    for n in n_q_tiles:
        cu_q_tiles.append(cu_q_tiles[-1] + n)
    total_q_tiles = cu_q_tiles[-1]

    cu_list, idx_list = [0], []
    seg_selected = []  # record which blocks each segment selected, for the reference mask / zero-gradient check
    for _h_kv in range(nheads_kv):
        for b in range(batch):
            sq, sk = seqlens_q[b], seqlens_k[b]
            prefix = sk - sq
            n_k_blocks = (sk + K_BLOCK_N - 1) // K_BLOCK_N
            for m in range(n_q_tiles[b]):
                q_start = m_block_to_q_pos(m, K_BLOCK_M, gqa)
                if q_start >= sq:
                    cu_list.append(len(idx_list))
                    seg_selected.append(set())
                    continue
                q_k_blk = q_pos_to_k_block(q_start, prefix, K_BLOCK_N)
                causal_max = min(q_k_blk + 1, n_k_blocks)
                sel = set(range(min(sink, causal_max)))
                sel.update(range(max(0, q_k_blk - window + 1), causal_max))
                sel = sorted(sel)
                idx_list.extend(sel)
                cu_list.append(len(idx_list))
                seg_selected.append(set(sel))

    block_sparse_cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
    block_sparse_idx = torch.tensor(idx_list, dtype=torch.int32, device=device)
    cu_q_tiles_t = torch.tensor(cu_q_tiles, dtype=torch.int32, device=device)
    return block_sparse_cu, block_sparse_idx, total_q_tiles, cu_q_tiles_t, seg_selected


def build_ref_mask(seqlens_q, seqlens_k, nheads_q, nheads_kv, seg_selected,
                   total_q_tiles, cu_q_tiles, device="cuda"):
    """mask[b, h_q, q_pos, k_pos] = CSR-selected AND bottom-right causal."""
    gqa = nheads_q // nheads_kv
    batch = len(seqlens_q)
    masks = []
    for b in range(batch):
        sq, sk = seqlens_q[b], seqlens_k[b]
        prefix = sk - sq
        mask_b = torch.zeros(nheads_q, sq, sk, dtype=torch.bool, device=device)
        for h_q in range(nheads_q):
            h_kv, hin = h_q // gqa, h_q % gqa
            for q_pos in range(sq):
                p = (q_pos * gqa + hin) // K_BLOCK_M
                seg = h_kv * total_q_tiles + cu_q_tiles[b] + p
                for blk in seg_selected[seg]:
                    k0 = blk * K_BLOCK_N
                    k1 = min(k0 + K_BLOCK_N, sk)
                    mask_b[h_q, q_pos, k0:k1] = True
                causal_limit = q_pos + prefix  # allow k_pos <= causal_limit
                if causal_limit + 1 < sk:
                    mask_b[h_q, q_pos, causal_limit + 1:] = False
        masks.append(mask_b)
    return masks


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def run_case(dtype, nheads_q, nheads_kv, seqlens_q, seqlens_k, sink, window, seed, tag,
             head_dim=HEAD_DIM):
    torch.manual_seed(seed)
    device = "cuda"
    gqa = nheads_q // nheads_kv
    cu_q = torch.tensor([0] + list(torch.tensor(seqlens_q).cumsum(0)), dtype=torch.int32, device=device)
    cu_k = torch.tensor([0] + list(torch.tensor(seqlens_k).cumsum(0)), dtype=torch.int32, device=device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    max_q, max_k = max(seqlens_q), max(seqlens_k)

    q = (torch.randn(total_q, nheads_q, head_dim, dtype=torch.float32, device=device) * 0.5).to(dtype)
    k = (torch.randn(total_k, nheads_kv, head_dim, dtype=torch.float32, device=device) * 0.5).to(dtype)
    v = (torch.randn(total_k, nheads_kv, head_dim, dtype=torch.float32, device=device) * 0.5).to(dtype)
    grad_out = torch.randn(total_q, nheads_q, head_dim, dtype=torch.float32, device=device) * 0.5

    bs_cu, bs_idx, total_q_tiles, cu_q_tiles, seg_selected = build_packgqa_csr_varlen(
        seqlens_q, seqlens_k, nheads_q, nheads_kv, sink, window, device)

    qk = q.detach().clone().requires_grad_(True)
    kk = k.detach().clone().requires_grad_(True)
    vk = v.detach().clone().requires_grad_(True)
    out = flash_attn_varlen_func(
        qk, kk, vk, cu_q, cu_k, max_q, max_k, causal=True,
        block_sparse_cu=bs_cu, block_sparse_idx=bs_idx,
        total_q_tiles=total_q_tiles, cu_q_tiles=cu_q_tiles)
    (out.float() * grad_out).sum().backward()

    # fp32 masked reference
    masks = build_ref_mask(seqlens_q, seqlens_k, nheads_q, nheads_kv,
                           seg_selected, total_q_tiles, cu_q_tiles.tolist(), device)
    scale = head_dim ** -0.5
    dq_ref = torch.zeros_like(q, dtype=torch.float32)
    dk_ref = torch.zeros_like(k, dtype=torch.float32)
    dv_ref = torch.zeros_like(v, dtype=torch.float32)
    out_ref = torch.zeros_like(q, dtype=torch.float32)
    for b in range(len(seqlens_q)):
        q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
        k0, k1 = int(cu_k[b]), int(cu_k[b + 1])
        qb = q[q0:q1].float().requires_grad_(True)
        kb = k[k0:k1].float().requires_grad_(True)
        vb = v[k0:k1].float().requires_grad_(True)
        kexp = kb.repeat_interleave(gqa, dim=1)
        vexp = vb.repeat_interleave(gqa, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qb * scale, kexp)
        scores = scores.masked_fill(~masks[b], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        ob = torch.einsum("hqk,khd->qhd", p, vexp)
        dqb, dkb, dvb = torch.autograd.grad(
            (ob * grad_out[q0:q1]).sum(), [qb, kb, vb])
        out_ref[q0:q1] = ob.detach()
        dq_ref[q0:q1] = dqb
        dk_ref[k0:k1] = dkb
        dv_ref[k0:k1] = dvb

    e_out = rel_l2(out, out_ref)
    e_dq = rel_l2(qk.grad, dq_ref)
    e_dk = rel_l2(kk.grad, dk_ref)
    e_dv = rel_l2(vk.grad, dv_ref)

    # Gradients at unselected K positions must be 0
    zero_viol = 0
    for b in range(len(seqlens_q)):
        sk = seqlens_k[b]
        n_k_blocks = (sk + K_BLOCK_N - 1) // K_BLOCK_N
        k0, k1 = int(cu_k[b]), int(cu_k[b + 1])
        for h_kv in range(nheads_kv):
            selected = set()
            base = h_kv * total_q_tiles + cu_q_tiles[b]
            n_q_tiles_b = (seqlens_q[b] * gqa + K_BLOCK_M - 1) // K_BLOCK_M
            for m in range(n_q_tiles_b):
                selected |= seg_selected[base + m]
            for blk in range(n_k_blocks):
                if blk not in selected:
                    s, e = k0 + blk * K_BLOCK_N, min(k0 + (blk + 1) * K_BLOCK_N, k1)
                    zero_viol += int(kk.grad[s:e, h_kv].abs().max().item() > 0)
                    zero_viol += int(vk.grad[s:e, h_kv].abs().max().item() > 0)

    thr_out = 0.02 if dtype == torch.bfloat16 else 0.01
    thr_grad = 0.03 if dtype == torch.bfloat16 else 0.02
    ok = (e_out < thr_out and e_dq < thr_grad and e_dk < thr_grad
          and e_dv < thr_grad and zero_viol == 0)
    status = "OK" if ok else "FAIL"
    print(f"{status} {tag}: out={e_out:.4e} dq={e_dq:.4e} dk={e_dk:.4e} "
          f"dv={e_dv:.4e} zero_viol={zero_viol}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available")
        return
    all_ok = True
    cases = [
        # (dtype, h_q, h_kv, seqlens_q, seqlens_k, sink, window, seed, tag)
        (torch.bfloat16, 8, 2, [128, 37], [128, 96], 1, 2, 0, "bf16-g4-varlen"),
        (torch.float16, 8, 2, [128, 37], [128, 96], 1, 2, 1, "fp16-g4-varlen"),
        (torch.bfloat16, 8, 8, [100, 64], [100, 160], 1, 2, 2, "bf16-g1-mha"),
        (torch.bfloat16, 7, 1, [128, 55], [128, 128], 1, 2, 3, "bf16-g7"),
        (torch.bfloat16, 12, 1, [100, 33], [100, 77], 1, 2, 4, "bf16-g12"),
        (torch.bfloat16, 8, 2, [64], [512], 1, 2, 5, "bf16-longprefix-unselected"),
        (torch.float16, 8, 2, [256, 129], [256, 300], 2, 3, 6, "fp16-g4-sink2-win3"),
        # hdim256
        (torch.bfloat16, 8, 2, [128, 37], [128, 96], 1, 2, 10, "bf16-g4-hdim256", 256),
        (torch.bfloat16, 8, 8, [100, 64], [100, 160], 1, 2, 11, "bf16-g1-hdim256", 256),
        (torch.bfloat16, 12, 1, [100, 33], [100, 77], 1, 2, 12, "bf16-g12-hdim256", 256),
    ]
    for case in cases:
        all_ok &= run_case(*case)
    print("ALL PASS" if all_ok else "SOME CASES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
