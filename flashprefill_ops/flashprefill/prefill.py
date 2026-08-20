from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import triton

from .flash_attn_interface import flash_attn_with_kvcache
from flash_block_sparse_index_triton import (
    SparseIndexWorkspace,
    build_block_sparse_index_fast,
)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SparseIndex:
    block_sparse_cu: torch.Tensor
    block_sparse_idx: torch.Tensor
    total_q_tiles: int
    cu_q_tiles: torch.Tensor
    # Block means for the zero-order correction of unselected blocks
    # (populated only when use_mean_correction is enabled).
    k_mean: Optional[torch.Tensor] = None
    v_mean: Optional[torch.Tensor] = None


class FlashPrefill:
    """Index-select + block-sparse prefill pipeline.

    Index selection uses a reusable workspace fast path isolated per CUDA
    stream. Workspaces are capacity matched, so smaller dynamic shapes reuse a
    previously allocated larger buffer without allocator or host-sync overhead.
    Set FLASHPREFILL_DEBUG=1 to print selected-block density and sparsity.
    Set FLASHPREFILL_FULL_CAUSAL_INDEX=1 to bypass index selection and pass a
    null sparse index, selecting the full-attention kernel path.
    """

    def __init__(
        self,
        *,
        k_block_m: int = 128,
        k_block_n: int = 64,
        abs_threshold: float = 1.0,
        attention_sink: int = 2,
        window_size: int = 4,
        last_n_blocks: int = 2,
        min_sparse_q_len: int = 0,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        num_splits: int = 1,
        debug_env: str = "FLASHPREFILL_DEBUG",
        max_cached_workspaces: int = 32,
        use_mean_correction: bool = False,
    ) -> None:
        if min_sparse_q_len < 0:
            raise ValueError("min_sparse_q_len must be non-negative")
        if abs_threshold < 0.0:
            raise ValueError("abs_threshold must be non-negative")
        if max_cached_workspaces <= 0:
            raise ValueError("max_cached_workspaces must be positive")
        if k_block_n % 64 != 0:
            raise ValueError("k_block_n must be a multiple of 64 (the attention kernel tile size)")
        self.k_block_m = k_block_m
        self.k_block_n = k_block_n
        # Physical 64-token attention tiles per logical selection block.
        self.n_sub = k_block_n // 64
        # Zero-order correction (Sol-Attn style): unselected blocks contribute
        # their pooled K/V means inside the attention kernel epilogue (fwd only).
        self.use_mean_correction = use_mean_correction
        # Max-based dynamic threshold (the paper's alpha): a block's score is
        # the tile-level softmax energy summed over the packed rows,
        #   S[b] = sum_rows 2^(qk[row, b] - max logit of the tile),
        # accumulated online in the single scoring pass; keep blocks with
        # S[b] >= abs_threshold * max_b S[b]. alpha in (0, 1]; larger is
        # sparser. Paper-calibrated values (block size 128): Llama-3.1-8B
        # 0.18, Qwen2.5-7B 0.08, Qwen3-30B 0.12.
        self.abs_threshold = abs_threshold
        self.attention_sink = attention_sink
        self.window_size = window_size
        self.last_n_blocks = last_n_blocks
        self.min_sparse_q_len = min_sparse_q_len
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.num_splits = num_splits
        self.debug_env = debug_env
        self.max_cached_workspaces = max_cached_workspaces
        self._workspaces: OrderedDict[int, SparseIndexWorkspace] = OrderedDict()

    @staticmethod
    def _host_lengths(
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        q_lens: Optional[Sequence[int]],
        max_cache_seqlen: Optional[int],
    ) -> tuple[tuple[int, ...], int]:
        if q_lens is None:
            q_lens_tuple = tuple(
                int(x) for x in (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
            )
        else:
            q_lens_tuple = tuple(int(x) for x in q_lens)
        batch_size = cache_seqlens.numel()
        if len(q_lens_tuple) != batch_size:
            raise ValueError("q_lens must contain one length per request")
        if max_cache_seqlen is None:
            max_k_len = int(cache_seqlens.max().item()) if batch_size else 0
        else:
            max_k_len = int(max_cache_seqlen)
        return q_lens_tuple, max_k_len

    def _get_workspace(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        q_lens: Sequence[int],
        max_k_len: int,
        v_cache: Optional[torch.Tensor] = None,
    ) -> SparseIndexWorkspace:
        batch_size = len(q_lens)
        num_q_heads = q.shape[1]
        num_kv_heads = k_cache.shape[2]
        head_dim = q.shape[2]
        head_dim_v = head_dim if v_cache is None else v_cache.shape[-1]
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if max_k_len < 0:
            raise ValueError("max_cache_seqlen must be non-negative")
        gqa_ratio = num_q_heads // num_kv_heads
        q_tiles = tuple(
            (q_len * gqa_ratio + self.k_block_m - 1) // self.k_block_m
            for q_len in q_lens
        )
        total_q_tiles = sum(q_tiles)
        max_q_tiles = max(q_tiles, default=0)
        max_k_blocks = triton.cdiv(max_k_len, self.k_block_n)
        if total_q_tiles == 0 or max_k_blocks == 0:
            raise ValueError("workspace is not required for an empty index")

        stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
        candidates = []
        for key, workspace in self._workspaces.items():
            compatible = (
                workspace.cu_q_tiles.device == q.device
                and workspace.k_mean.dtype == k_cache.dtype
                and workspace.num_kv_heads == num_kv_heads
                and workspace.head_dim == head_dim
                and workspace.n_sub == self.n_sub
                and workspace.use_mean_correction == self.use_mean_correction
                and workspace.head_dim_v == head_dim_v
                and workspace.batch_capacity >= batch_size
                and workspace.total_q_tiles_capacity >= total_q_tiles
                and workspace.max_q_tiles_capacity >= max_q_tiles
                and workspace.max_k_blocks_capacity >= max_k_blocks
                and key >> 32 == stream_id
            )
            if compatible:
                waste = (
                    workspace.batch_capacity - batch_size,
                    workspace.total_q_tiles_capacity - total_q_tiles,
                    workspace.max_k_blocks_capacity - max_k_blocks,
                )
                candidates.append((waste, key, workspace))

        if candidates:
            _, key, workspace = min(candidates, key=lambda item: item[0])
            self._workspaces.move_to_end(key)
        else:
            serial = 0
            key = stream_id << 32
            while key + serial in self._workspaces:
                serial += 1
            key += serial
            workspace = SparseIndexWorkspace(
                batch_size=batch_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                total_q_tiles=total_q_tiles,
                max_q_tiles=max_q_tiles,
                max_k_blocks=max_k_blocks,
                dtype=k_cache.dtype,
                device=q.device,
                cu_q_tiles=torch.empty(
                    batch_size + 1, dtype=torch.int32, device=q.device
                ),
                n_sub=self.n_sub,
                use_mean_correction=self.use_mean_correction,
                head_dim_v=head_dim_v,
            )
            self._workspaces[key] = workspace
            while len(self._workspaces) > self.max_cached_workspaces:
                self._workspaces.popitem(last=False)

        workspace.activate(
            batch_size=batch_size,
            total_q_tiles=total_q_tiles,
            max_q_tiles=max_q_tiles,
            max_k_blocks=max_k_blocks,
        )
        return workspace

    def clear_workspaces(self) -> None:
        self._workspaces.clear()

    def index_select(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        v_cache: Optional[torch.Tensor] = None,
        q_lens: Optional[Sequence[int]] = None,
        max_cache_seqlen: Optional[int] = None,
        q_descale: float = 1.0,
        k_descale: float = 1.0,
        softmax_scale: Optional[float] = None,
    ) -> SparseIndex:
        if self.use_mean_correction and v_cache is None:
            raise ValueError("v_cache is required when use_mean_correction is enabled")
        host_q_lens, max_k_len = self._host_lengths(
            cu_seqlens_q, cache_seqlens, q_lens, max_cache_seqlen
        )
        if any(q_len < 0 for q_len in host_q_lens):
            raise ValueError("q_lens must be non-negative")
        if sum(host_q_lens) != q.shape[0]:
            raise ValueError("sum(q_lens) must equal q.shape[0]")
        if not host_q_lens or sum(host_q_lens) == 0 or max_k_len == 0:
            num_kv_heads = k_cache.shape[2]
            gqa_ratio = q.shape[1] // num_kv_heads
            q_tiles = tuple(
                (q_len * gqa_ratio + self.k_block_m - 1) // self.k_block_m
                for q_len in host_q_lens
            )
            cu_q_tiles = torch.tensor(
                (0, *torch.tensor(q_tiles).cumsum(0).tolist()),
                dtype=torch.int32,
                device=q.device,
            )
            total_q_tiles = sum(q_tiles)
            result = SparseIndex(
                torch.zeros(
                    num_kv_heads * total_q_tiles + 1,
                    dtype=torch.int32,
                    device=q.device,
                ),
                torch.empty(0, dtype=torch.int32, device=q.device),
                total_q_tiles,
                cu_q_tiles,
            )
        else:
            workspace = self._get_workspace(
                q, k_cache, host_q_lens, max_k_len, v_cache
            )
            cu, idx, total_q_tiles, cu_q_tiles = build_block_sparse_index_fast(
                q,
                k_cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                workspace,
                v_cache=v_cache,
                k_block_m=self.k_block_m,
                k_block_n=self.k_block_n,
                abs_threshold=self.abs_threshold,
                attention_sink=self.attention_sink,
                window_size=self.window_size,
                last_n_blocks=self.last_n_blocks,
                min_sparse_q_len=self.min_sparse_q_len,
                causal=self.causal,
                softmax_scale=(
                    self.softmax_scale if softmax_scale is None else softmax_scale
                ),
                q_descale=q_descale,
                k_descale=k_descale,
            )
            result = SparseIndex(
                cu,
                idx,
                total_q_tiles,
                cu_q_tiles,
                k_mean=workspace.k_mean if self.use_mean_correction else None,
                v_mean=workspace.v_mean if self.use_mean_correction else None,
            )
        if _env_enabled(self.debug_env):
            host_q_lens = (
                tuple(int(x) for x in q_lens)
                if q_lens is not None
                else tuple(
                    int(x)
                    for x in (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
                )
            )
            self._print_sparsity(
                result, host_q_lens, cache_seqlens, q.shape[1], k_cache.shape[2]
            )
        return result

    def _print_sparsity(
        self,
        index: SparseIndex,
        q_lens: Sequence[int],
        cache_seqlens: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> None:
        kv_lens = tuple(int(x) for x in cache_seqlens.tolist())
        q_tile_counts = tuple(
            int(x)
            for x in (index.cu_q_tiles[1:] - index.cu_q_tiles[:-1]).tolist()
        )
        gqa_ratio = num_q_heads // num_kv_heads
        # selected (block_sparse_cu) is in physical 64-token tile units after
        # logical-to-physical expansion, so candidates must be counted in
        # 64-token tiles too.
        tile_n = 64
        possible = 0
        for q_len, kv_len, tile_count in zip(q_lens, kv_lens, q_tile_counts):
            prefix = kv_len - q_len
            n_k_blocks = (kv_len + tile_n - 1) // tile_n
            for tile in range(tile_count):
                packed_end = min((tile + 1) * self.k_block_m, q_len * gqa_ratio)
                q_last = (packed_end - 1) // gqa_ratio
                if self.causal:
                    possible += min(
                        (prefix + q_last) // tile_n + 1, n_k_blocks
                    )
                else:
                    possible += n_k_blocks
        possible *= num_kv_heads
        selected = int(index.block_sparse_cu[-1].item())
        density = 100.0 * selected / possible if possible else 0.0
        print(
            f"[FlashPrefill] selected={selected}/{possible}, "
            f"density={density:.2f}%, sparsity={100.0 - density:.2f}%"
        )

    def block_sparse_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        index: Optional[SparseIndex],
        *,
        max_seqlen_q: int,
        softmax_scale: Optional[float] = None,
        **attention_kwargs: Any,
    ) -> torch.Tensor:
        reserved = {
            "block_sparse_cu", "block_sparse_idx", "total_q_tiles", "cu_q_tiles",
            "k_mean", "v_mean", "mean_k_block_size",
        }
        overlap = reserved.intersection(attention_kwargs)
        if overlap:
            raise TypeError(f"sparse index arguments are managed internally: {sorted(overlap)}")
        with_correction = (
            index is not None and index.k_mean is not None and index.v_mean is not None
        )
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=(
                self.softmax_scale if softmax_scale is None else softmax_scale
            ),
            causal=self.causal,
            num_splits=self.num_splits,
            block_sparse_cu=None if index is None else index.block_sparse_cu,
            block_sparse_idx=None if index is None else index.block_sparse_idx,
            total_q_tiles=None if index is None else index.total_q_tiles,
            cu_q_tiles=None if index is None else index.cu_q_tiles,
            k_mean=index.k_mean if with_correction else None,
            v_mean=index.v_mean if with_correction else None,
            mean_k_block_size=self.k_block_n if with_correction else 0,
            **attention_kwargs,
        )

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        q_lens: Optional[Sequence[int]] = None,
        max_cache_seqlen: Optional[int] = None,
        q_descale: float = 1.0,
        k_descale: float = 1.0,
        softmax_scale: Optional[float] = None,
        attention_k_descale: Optional[torch.Tensor] = None,
        **attention_kwargs: Any,
    ) -> torch.Tensor:
        host_q_lens, inferred_max_k = self._host_lengths(
            cu_seqlens_q, cache_seqlens, q_lens, max_cache_seqlen
        )
        index = None
        if not _env_enabled("FLASHPREFILL_FULL_CAUSAL_INDEX"):
            index = self.index_select(
                q,
                k_cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                v_cache=v_cache,
                q_lens=host_q_lens,
                max_cache_seqlen=inferred_max_k,
                q_descale=q_descale,
                k_descale=k_descale,
                softmax_scale=softmax_scale,
            )
        return self.block_sparse_attention(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            index,
            max_seqlen_q=max(host_q_lens, default=0),
            softmax_scale=softmax_scale,
            k_descale=attention_k_descale,
            **attention_kwargs,
        )


__all__ = ["FlashPrefill", "SparseIndex"]