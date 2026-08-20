"""
Comparison script: compare the accuracy difference between FP8 and BF16 under various masks.
FP8 mode takes q_descale, k_descale, v_descale for dequantization compensation.

Tested mask patterns:
  1. Dense Causal
  2. Dense Causal + Sliding Window
  3. Block Sparse Causal
  4. Block Sparse Causal + Sliding Window

Output: max_abs_err, mean_abs_err, cos_sim of FP8 vs BF16 for each configuration.
"""
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_block_sparse import (
    get_tile_sizes, create_test_inputs,
    m_block_to_q_pos, q_pos_to_k_block,
)
from flash_attn_interface import flash_attn_with_kvcache
from bench_flashprefill_lastn2 import build_index_packgqa

torch.manual_seed(42)
device = "cuda"

FP8_E4M3_MAX = 448.0  # float8_e4m3fn max representable value


def quantize_to_fp8(x_fp32, batch_size, nheads_kv):
    """Per-(batch, kv_head) symmetric quantization to fp8_e4m3fn.
    x_fp32 shape: (total_q, nheads, head_dim) for Q or (num_pages, page_size, nheads_kv, head_dim) for KV.
    Returns: (x_fp8, descale) where descale shape is (batch_size, nheads_kv).
    """
    raise NotImplementedError("Use quantize_q / quantize_kv instead")


def quantize_q(q_fp32, seqlen_q, batch_size, nheads_q, nheads_kv, head_dim):
    """Quantize Q tensor to fp8 with per-(batch, kv_head) scaling.
    Q shape: (total_q, nheads_q, head_dim) flattened as (batch * seqlen, nheads, dim).
    Returns: (q_fp8, q_descale) where q_descale shape (batch, nheads_kv).
    """
    gqa_ratio = nheads_q // nheads_kv
    # Reshape: (batch, seqlen_q, nheads_q, head_dim) -> (batch, seqlen_q, nheads_kv, gqa_ratio, head_dim)
    q_4d = q_fp32.reshape(batch_size, seqlen_q, nheads_q, head_dim)
    q_grouped = q_4d.reshape(batch_size, seqlen_q, nheads_kv, gqa_ratio, head_dim)

    # Per (batch, kv_head) max abs
    amax = q_grouped.abs().amax(dim=(1, 3, 4))  # (batch, nheads_kv)
    scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
    q_descale = scale.squeeze(-1) if scale.dim() > 2 else scale  # (batch, nheads_kv)

    # Quantize
    scale_expanded = scale.unsqueeze(1).unsqueeze(3)  # (batch, 1, nheads_kv, 1, gqa_ratio, head_dim)... need careful
    # Actually: scale shape (batch, nheads_kv), need to broadcast over (seqlen, gqa_ratio, head_dim)
    scale_b = scale.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (batch, 1, nheads_kv, 1, 1)
    q_scaled = (q_grouped / scale_b).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q_fp8 = q_scaled.reshape(batch_size, seqlen_q, nheads_q, head_dim).reshape(-1, nheads_q, head_dim).to(torch.float8_e4m3fn)
    return q_fp8, q_descale


def quantize_kv(kv_fp32, batch_size, nheads_kv, head_dim, seqlen_k, num_pages, page_size):
    """Quantize K or V cache to fp8 with per-(batch, kv_head) scaling.
    KV cache shape: (num_pages, page_size, nheads_kv, head_dim).
    page_table maps batch * seqlen_k pages sequentially (pages 0..batch*seqlen_k-1 used).
    Returns: (kv_fp8, kv_descale) where kv_descale shape (batch, nheads_kv).
    """
    used_pages = batch_size * seqlen_k
    # Only quantize the pages that are actually referenced by page_table
    kv_used = kv_fp32[:used_pages]  # (used_pages, page_size, nheads_kv, head_dim)
    kv_4d = kv_used.reshape(batch_size, seqlen_k, nheads_kv, head_dim)
    amax = kv_4d.abs().amax(dim=(1, 3))  # (batch, nheads_kv)
    scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
    kv_descale = scale

    scale_b = scale.unsqueeze(1).unsqueeze(3)  # (batch, 1, nheads_kv, 1)
    kv_scaled = (kv_4d / scale_b).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    # Write back into full cache (unused pages left as-is in fp8)
    kv_out = kv_fp32.clone()
    kv_out[:used_pages] = kv_scaled.reshape(used_pages, page_size, nheads_kv, head_dim).to(torch.float8_e4m3fn)
    # Convert unused pages too (they won't be accessed but tensor dtype must be uniform)
    kv_out = kv_out.to(torch.float8_e4m3fn)
    return kv_out, kv_descale


def create_dual_inputs(batch_size, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim, device):
    """Create identical random inputs, return both bf16 and fp8 versions with descales."""
    total_q = batch_size * seqlen_q
    page_size = 1
    num_pages = batch_size * seqlen_k * 2
    scale = 0.5

    # Shared fp32 source
    q_fp32 = torch.randn(total_q, nheads_q, head_dim, dtype=torch.float32, device=device)
    k_fp32 = torch.randn(num_pages, page_size, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale
    v_fp32 = torch.randn(num_pages, page_size, nheads_kv, head_dim, dtype=torch.float32, device=device) * scale

    # BF16
    q_bf16 = q_fp32.to(torch.bfloat16)
    k_bf16 = k_fp32.to(torch.bfloat16)
    v_bf16 = v_fp32.to(torch.bfloat16)

    # FP8
    q_fp8, q_descale = quantize_q(q_fp32, seqlen_q, batch_size, nheads_q, nheads_kv, head_dim)
    k_fp8, k_descale = quantize_kv(k_fp32, batch_size, nheads_kv, head_dim, seqlen_k, num_pages, page_size)
    v_fp8, v_descale = quantize_kv(v_fp32, batch_size, nheads_kv, head_dim, seqlen_k, num_pages, page_size)

    # Shared
    page_table = torch.zeros(batch_size, seqlen_k, dtype=torch.int32, device=device)
    for b in range(batch_size):
        page_table[b] = torch.arange(b * seqlen_k, (b + 1) * seqlen_k, dtype=torch.int32, device=device)
    cache_seqlens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0] + [seqlen_q * (i+1) for i in range(batch_size)], dtype=torch.int32, device=device)

    bf16_inputs = dict(q=q_bf16, k_cache=k_bf16, v_cache=v_bf16,
                       page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q)
    fp8_inputs = dict(q=q_fp8, k_cache=k_fp8, v_cache=v_fp8,
                      q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                      page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q)
    return bf16_inputs, fp8_inputs


def run_attention(inputs, seqlen_q, head_dim, softmax_scale,
                  causal=True, window_size=(-1, -1), block_sparse=None, num_splits=1):
    kwargs = dict(
        q=inputs["q"], k_cache=inputs["k_cache"], v_cache=inputs["v_cache"],
        page_table=inputs["page_table"], cache_seqlens=inputs["cache_seqlens"],
        cu_seqlens_q=inputs["cu_seqlens_q"], max_seqlen_q=seqlen_q,
        softmax_scale=softmax_scale, causal=causal,
        window_size=window_size,
        num_splits=num_splits,
    )
    # FP8 descale
    if "q_descale" in inputs:
        kwargs["q_descale"] = inputs["q_descale"]
        kwargs["k_descale"] = inputs["k_descale"]
        kwargs["v_descale"] = inputs["v_descale"]
    # Block sparse
    if block_sparse is not None:
        kwargs["block_sparse_cu"] = block_sparse["cu"]
        kwargs["block_sparse_idx"] = block_sparse["idx"]
        kwargs["total_q_tiles"] = block_sparse["total_q_tiles"]
        kwargs["cu_q_tiles"] = block_sparse["cu_q_tiles"]
    return flash_attn_with_kvcache(**kwargs)


def compute_errors(out_fp8, out_bf16):
    """Compare FP8 output against BF16 reference."""
    out_bf16_f = out_bf16.to(torch.float32)
    out_fp8_f = out_fp8.to(torch.float32)
    abs_diff = (out_fp8_f - out_bf16_f).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    median_abs_err = abs_diff.median().item()
    # Cosine similarity
    dot = (out_bf16_f * out_fp8_f).sum(dim=-1)
    norm_bf16 = out_bf16_f.norm(dim=-1)
    norm_fp8 = out_fp8_f.norm(dim=-1)
    cos_sim = (dot / (norm_bf16 * norm_fp8 + 1e-12)).mean().item()
    # Relative error (mean of per-element relative errors where ref is significant)
    ref_abs = out_bf16_f.abs()
    rel_mask = ref_abs > 1e-3
    rel_err = (abs_diff[rel_mask] / ref_abs[rel_mask]).mean().item() if rel_mask.any() else 0.0
    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "median_abs_err": median_abs_err,
        "cos_sim": cos_sim,
        "rel_err": rel_err,
    }


def main():
    BATCH = 2
    SEQLEN_Q = 1024
    SEQLEN_K = 1024
    NHEADS_Q = 8
    NHEADS_KV = 2
    HEAD_DIM = 128
    NUM_SPLITS = 1

    element_size = 1
    kBlockM, kBlockN = get_tile_sizes(HEAD_DIM, element_size, is_causal=True, paged_kv_non_tma=True)
    softmax_scale = HEAD_DIM ** (-0.5)

    print(f"FP8 vs BF16 accuracy comparison")
    print(f"  batch={BATCH}, seqlen_q={SEQLEN_Q}, seqlen_k={SEQLEN_K}")
    print(f"  nheads_q={NHEADS_Q}, nheads_kv={NHEADS_KV}, head_dim={HEAD_DIM}")
    print(f"  kBlockM={kBlockM}, kBlockN={kBlockN}")
    print(f"  softmax_scale={softmax_scale:.6f}")
    print(f"{'='*100}")

    # Create shared inputs
    bf16_inputs, fp8_inputs = create_dual_inputs(
        BATCH, SEQLEN_Q, SEQLEN_K, NHEADS_Q, NHEADS_KV, HEAD_DIM, device)

    # Build block sparse indices with different sparsity levels
    sparse_configs = [
        ("Sparse (rand=0, sink1,win2,ln2)",  0),
        ("Sparse (rand=4, sink1,win2,ln2)",  4),
        ("Sparse (rand=8, sink1,win2,ln2)",  8),
        ("Sparse (rand=16,sink1,win2,ln2)", 16),
    ]
    sparse_indices = []
    for name, rand_blocks in sparse_configs:
        cu, idx, tqt, cqt = build_index_packgqa(
            BATCH, NHEADS_Q, NHEADS_KV, SEQLEN_Q, SEQLEN_K,
            kBlockM, kBlockN,
            attention_sink=1, window=2, last_n_blocks=2,
            num_random_blocks=rand_blocks, causal=True, device=device)
        sparse_indices.append((name, dict(cu=cu, idx=idx, total_q_tiles=tqt, cu_q_tiles=cqt)))

    # Define test configurations: (name, causal, window, block_sparse_dict_or_None)
    configs = [
        ("Dense Causal", True,  None),
    ]
    for sname, sdict in sparse_indices:
        configs.append((sname, True, sdict))

    results = []
    for name, causal, bs in configs:
        try:
            out_bf16 = run_attention(
                bf16_inputs, SEQLEN_Q, HEAD_DIM, softmax_scale,
                causal=causal, window_size=(-1, -1),
                block_sparse=bs, num_splits=NUM_SPLITS)
            out_fp8 = run_attention(
                fp8_inputs, SEQLEN_Q, HEAD_DIM, softmax_scale,
                causal=causal, window_size=(-1, -1),
                block_sparse=bs, num_splits=NUM_SPLITS)

            errs = compute_errors(out_fp8, out_bf16)
            results.append((name, errs, None))
            print(f"  {name:<28} | max_abs={errs['max_abs_err']:.6f}  mean_abs={errs['mean_abs_err']:.6f}  "
                  f"median_abs={errs['median_abs_err']:.6f}  cos_sim={errs['cos_sim']:.6f}  rel_err={errs['rel_err']:.4f}")
        except Exception as e:
            results.append((name, None, str(e)))
            print(f"  {name:<28} | ERROR: {str(e)[:100]}")
        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("Notes:")
    print("  - BF16 is the reference; FP8 uses per-(batch, kv_head) symmetric quantization + descale dequantization compensation")
    print("  - cos_sim closer to 1.0 is better; smaller max_abs / mean_abs is better")
    print(f"{'='*100}")

    return results


if __name__ == "__main__":
    main()
