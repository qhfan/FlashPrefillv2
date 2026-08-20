"""
FlashPrefill attention backend for SGLang.

This backend integrates the block-sparse prefill operator from the standalone
``flashprefill`` package (a FlashAttention-3 fork with block-sparse attention)
into SGLang's runtime.

Design (mirrors FA3's paged metadata + delegate everything non-prefill to FA3):

  1. Block-sparse scoring / selection  (``FlashPrefill.index_select``)
     builds a CSR compact index of selected KV blocks per (kv_head, packed Q tile).
  2. Sparse flash attention             (``FlashPrefill.block_sparse_attention``)
     consumes the index together with the paged KV cache.

The two steps are fused into ``FlashPrefill.__call__`` (one-shot pipeline).
The ``FlashPrefill`` instance stores immutable selection configuration and is
created ONCE in ``__init__``; each forward rebuilds its sparse index directly.

SCOPE: The block-sparse method only applies to the PREFILL / EXTEND phase, where
sequences are long and skipping key-blocks actually saves compute. DECODE issues
exactly one query token per step, so block selection brings no benefit; decode is
therefore delegated to a standard full-attention backend (FA3) untouched.

  - forward_extend  -> block-sparse prefill (select blocks + sparse flash)
  - forward_decode  -> delegated verbatim to an internal FlashAttentionBackend

NOTE: k_block_m (128, Q-tile packed rows) is hard-coded to match the compiled
PackGQA kernel tile. k_block_n (logical K-block tokens) is configurable via
--flashprefill-k-block-n: it must be a power-of-two multiple of 64 (64/128/256/
512); each selected logical block is expanded into consecutive 64-token physical
tiles by the index builder, so the compiled attention kernel needs no rebuild.
The remaining selection hyper-params (abs_threshold, attention_sink,
window_size, last_n_blocks) also come from server CLI args.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


# ---------------------------------------------------------------------------
# FlashPrefill tile sizes. k_block_m MUST agree with the compiled PackGQA
# kernel tile (do NOT override). k_block_n is the LOGICAL selection block; the
# default 64 equals the physical attention tile, and larger values (128/256/
# 512) are supported by CSR expansion inside the index builder.
# ---------------------------------------------------------------------------
_FLASHPREFILL_K_BLOCK_M = 128  # Q-tile packed rows
_FLASHPREFILL_K_BLOCK_N = 64  # default logical K-block tokens


@dataclass
class FlashPrefillMetadata:
    """Per-forward metadata, built once and reused by every layer.

    Only the inputs the standalone ``FlashPrefill`` operator needs are kept here.
    All block-grid / CSR artifacts are produced inside the operator on each
    forward, so they are not pre-computed here.
    """

    # Real KV length per request (includes the already-cached prefix). int32 CUDA.
    cache_seqlens_int32: torch.Tensor = None
    # Max query length in this batch (number of *new* tokens for extend).
    max_seq_len_q: int = 1
    # Max KV length in this batch.
    max_seq_len_k: int = 0
    # Prefix-sum of *new* query tokens per request. shape (bs + 1,) int32 CUDA.
    cu_seqlens_q: torch.Tensor = None
    # Per-request Q lengths (host list). Used to avoid copying Q lengths to host.
    q_lens_list: list = None
    # Page-level page table: row r -> physical page ids for request r.
    # shape (bs, max_pages) int32 CUDA. max_pages = ceil(max_seq_len_k / page_size).
    page_table: torch.Tensor = None


class FlashPrefillAttnBackend(AttentionBackend):
    """Block-sparse prefill backend backed by the standalone ``flashprefill`` pkg."""

    def __init__(
        self,
        model_runner: "ModelRunner",
        attention_sink: int = 2,
        window: int = 4,
        abs_threshold: float = 1.0,
        last_n_blocks: int = 2,
        min_sparse_q_len: int = 4096,
        k_block_n: int = _FLASHPREFILL_K_BLOCK_N,
        use_mean_correction: bool = False,
        full_attention_layers: int = 0,
    ):
        super().__init__()
        if min_sparse_q_len < 0:
            raise ValueError("min_sparse_q_len must be non-negative")
        if full_attention_layers < 0:
            raise ValueError("full_attention_layers must be non-negative")

        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        # KV-cache dtype string ("auto" | "fp8_e4m3" | ...). Sourced exactly like
        # FA3 (`model_runner.server_args.kv_cache_dtype`). When != "auto" the paged
        # KV pool stores FP8 and the block-sparse kernels must descale K/V.
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        # torch dtype form (e.g. torch.float8_e4m3fn).
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        # Full-length upper bound, sourced exactly like FA3
        # (`self.max_context_len = model_runner.model_config.context_len`). Used to
        # tell a REAL sliding-window layer apart from a "pseudo-SWA" layer whose
        # window is set to >= the full context, which is numerically full
        # attention and must go through the block-sparse path.
        self.context_len = model_runner.model_config.context_len

        # Block-sparse hyper-params. k_block_m is hard-coded (kernel-coupled);
        # k_block_n and the selection knobs come from server CLI args
        # (--flashprefill-*).
        self.attention_sink = attention_sink
        self.window = window
        self.abs_threshold = abs_threshold
        self.last_n_blocks = last_n_blocks
        self.min_sparse_q_len = min_sparse_q_len
        self.full_attention_layers = full_attention_layers
        self.k_block_n = k_block_n

        self.forward_metadata: Optional[FlashPrefillMetadata] = None
        self.use_fa3_for_extend = False

        # The standalone FlashPrefill operator. ONE instance stores the immutable
        # selection configuration; sparse indices are rebuilt on every forward.
        from flashprefill import FlashPrefill

        self.flash_prefill = FlashPrefill(
            k_block_m=_FLASHPREFILL_K_BLOCK_M,
            k_block_n=self.k_block_n,
            abs_threshold=self.abs_threshold,
            attention_sink=self.attention_sink,
            window_size=self.window,
            last_n_blocks=self.last_n_blocks,
            min_sparse_q_len=self.min_sparse_q_len,
            causal=True,
            num_splits=1,
            use_mean_correction=use_mean_correction,
        )

        # Decode is NOT block-sparse: delegate it verbatim to a standard FA3
        # backend. This instance owns its own decode metadata and CUDA graph state.
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        self.decode_backend = FlashAttentionBackend(model_runner)

    # ------------------------------------------------------------------ #
    # Metadata                                                            #
    # ------------------------------------------------------------------ #
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build paged metadata once per forward.

        Decode delegates to the standard FA3 backend (no block-sparse). Only
        extend / prefill builds our own block-sparse metadata.
        """
        self.forward_metadata = None
        self.use_fa3_for_extend = False

        if forward_batch.forward_mode.is_decode_or_idle():
            self.decode_backend.init_forward_metadata(forward_batch)
            return

        if not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            raise NotImplementedError(
                f"FlashPrefill backend does not support mode {forward_batch.forward_mode}"
            )

        max_seq_len_q = int(max(forward_batch.extend_seq_lens_cpu))
        self.use_fa3_for_extend = max_seq_len_q <= self.min_sparse_q_len

        # Sparse attention is used when any request's query chunk is above the
        # threshold. Delegate to FA3 only when every query chunk is at or below it,
        # before allocating or copying any FlashPrefill metadata.
        if self.use_fa3_for_extend:
            self.decode_backend.init_forward_metadata(forward_batch)
            return

        # Long batches use FlashPrefill for ordinary full-attention layers, but
        # SWA layers still delegate to FA3 and require its metadata for the same
        # forward.
        self.decode_backend.init_forward_metadata(forward_batch)

        metadata = FlashPrefillMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        device = seqlens_in_batch.device
        max_seq_len_k = int(forward_batch.seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            # ---- EXTEND / CHUNKED PREFILL: q_len may differ from k_len ----
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = max_seq_len_k

            # ---- build the PAGE-LEVEL page table expected by FlashPrefill ----
            # sglang's req_to_token is a TOKEN-LEVEL mapping (logical_token ->
            # physical_slot). FlashPrefill wants [bs, max_pages] of physical PAGE
            # ids. Convert by striding every `page_size` tokens and dividing by
            # page_size (mirrors FA3's own page-table translation).
            token_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, :max_seq_len_k
            ]
            if self.page_size > 1:
                max_pages = (max_seq_len_k + self.page_size - 1) // self.page_size
                strided_indices = torch.arange(
                    0,
                    max_seq_len_k,
                    self.page_size,
                    device=device,
                    dtype=torch.long,
                )
                page_table = token_table[:, strided_indices] // self.page_size
            else:
                # page_size == 1: page id == physical slot id, no conversion needed.
                page_table = token_table
            metadata.page_table = page_table.to(torch.int32)

            if any(forward_batch.extend_prefix_lens_cpu):
                # Chunked prefill: there IS a cached prefix -> q_len < k_len.
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.q_lens_list = list(forward_batch.extend_seq_lens_cpu)
            else:
                # First / full prefill: q_len == k_len, reuse full seq_lens as q_lens.
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.q_lens_list = forward_batch.seq_lens_cpu.tolist()
        else:
            raise NotImplementedError(
                f"FlashPrefill backend does not support mode {forward_batch.forward_mode}"
            )

        self.forward_metadata = metadata

    # ------------------------------------------------------------------ #
    # Forward: extend (handles chunked prefill)                           #
    # ------------------------------------------------------------------ #
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        sinks = kwargs.get("sinks", None)

        # SWA (sliding-window) layers are NOT block-sparse: delegate them verbatim
        # to FA3. A window >= the full context length is NOT a real SWA layer;
        # we additionally require the window to be strictly smaller than the full
        # context.
        sw = getattr(layer, "sliding_window_size", None)
        is_swa_layer = sw is not None and 0 < sw < self.context_len
        if is_swa_layer:
            return self.decode_backend.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        # The first `full_attention_layers` layers stay dense (layer_id is
        # 0-based): delegate them to FA3 verbatim. Block sparsity only applies
        # from layer `full_attention_layers` onwards.
        if layer.layer_id < self.full_attention_layers:
            return self.decode_backend.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        # The batch-level decision was made before metadata construction. A short
        # batch therefore reaches this point with only FA3 metadata initialized.
        if self.use_fa3_for_extend:
            return self.decode_backend.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("FlashPrefill metadata was not initialized")

        # 1) Write this step's K/V into the paged pool (so the kernel can read it).
        if k is not None and save_kv_cache:
            cache_loc = forward_batch.out_cache_loc
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        if os.environ.get("FLASHPREFILL_LOG_SPARSITY") == "1":
            logger.warning(
                "[flashprefill] block-sparse path ENTERED: layer_id=%s q_shape=%s",
                getattr(layer, "layer_id", None),
                tuple(q.shape),
            )

        # 2) Fetch the paged KV buffers and reshape to the page layout FlashPrefill
        #    expects: [num_pages, page_size, num_kv_heads, head_dim].
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # 3) Match FA3's FP8 KV-cache semantics. Q is cast directly to the cache
        #    dtype (so its descale is 1), while K/V were divided by the layer
        #    scales when written to the paged cache and must be descaled on read.
        is_fp8_kv = self.kv_cache_dtype_str != "auto"
        index_k_descale = 1.0
        k_descale = None
        v_descale = None
        if is_fp8_kv:
            q = q.to(self.kv_cache_dtype)
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
                index_k_descale = getattr(layer, "k_scale_float", None)
                if index_k_descale is None:
                    index_k_descale = float(layer.k_scale.item())

        # if os.environ.get("FLASHPREFILL_DEBUG", "").strip().lower() in {
        #     "1",
        #     "true",
        #     "yes",
        #     "on",
        # }:
        #     import inspect
        #     import sys

        #     host_q_lens, debug_max_k_len = self.flash_prefill._host_lengths(
        #         metadata.cu_seqlens_q,
        #         metadata.cache_seqlens_int32,
        #         metadata.q_lens_list,
        #         metadata.max_seq_len_k,
        #     )
        #     debug_workspace = self.flash_prefill._get_workspace(
        #         q,
        #         key_cache,
        #         metadata.cache_seqlens_int32,
        #         host_q_lens,
        #         debug_max_k_len,
        #     )
        #     prefill_module = sys.modules[type(self.flash_prefill).__module__]
        #     index_builder = getattr(
        #         prefill_module, "build_block_sparse_index_fast", None
        #     )

        #     def tensor_info(tensor: torch.Tensor) -> str:
        #         return (
        #             f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
        #             f"stride={tensor.stride()} device={tensor.device} "
        #             f"contiguous={tensor.is_contiguous()}"
        #         )

        #     logger.warning(
        #         "[flashprefill-debug] layer_id=%s kv_cache_dtype_str=%s "
        #         "configured_kv_dtype=%s flashprefill_source=%s index_source=%s",
        #         getattr(layer, "layer_id", None),
        #         self.kv_cache_dtype_str,
        #         self.kv_cache_dtype,
        #         inspect.getsourcefile(type(self.flash_prefill)),
        #         inspect.getsourcefile(index_builder) if index_builder else None,
        #     )
        #     logger.warning(
        #         "[flashprefill-debug] q={%s} k_cache={%s} v_cache={%s} "
        #         "k_mean={%s}",
        #         tensor_info(q),
        #         tensor_info(key_cache),
        #         tensor_info(value_cache),
        #         tensor_info(debug_workspace.k_mean),
        #     )
        #     logger.warning(
        #         "[flashprefill-debug] page_table={%s} cache_seqlens={%s} "
        #         "cu_seqlens_q={%s} q_lens=%s max_q_len=%s max_k_len=%s",
        #         tensor_info(metadata.page_table),
        #         tensor_info(metadata.cache_seqlens_int32),
        #         tensor_info(metadata.cu_seqlens_q),
        #         host_q_lens,
        #         metadata.max_seq_len_q,
        #         debug_max_k_len,
        #     )

        # 4) Run sparse-index selection and block-sparse attention as one pipeline.
        #    Use the same layer-specific score scale and FP8 KV descales as FA3;
        #    the scalar K descale is also applied by the Triton selector.
        attention_kwargs = {}
        if sinks is not None:
            sinks = sinks.to(device=q.device, dtype=q.dtype).contiguous()
            attention_kwargs["sinks"] = sinks

        # With a non-none MoE a2a backend (e.g. deepep) and no DP attention,
        # require_attn_tp_gather() makes prepare_mlp_sync_batch pad the batch
        # to a token count aligned across the TP group. input_ids/positions are
        # padded, so q carries trailing pad rows that q_lens (from the unpadded
        # extend_seq_lens_cpu) does not describe, while flashprefill's
        # index_select requires sum(q_lens) == q.shape[0]. Slice the pad rows
        # off for the strict varlen pipeline and scatter the output back into
        # a padded buffer; downstream drops the pad rows anyway.
        real_q_tokens = sum(metadata.q_lens_list)
        q_sparse = q[:real_q_tokens] if q.shape[0] > real_q_tokens else q

        o = self.flash_prefill(
            q=q_sparse,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            q_lens=metadata.q_lens_list,
            max_cache_seqlen=metadata.max_seq_len_k,
            k_descale=index_k_descale,
            softmax_scale=layer.scaling,
            attention_k_descale=k_descale,
            v_descale=v_descale,
            **attention_kwargs,
        )
        o = o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        if o.shape[0] < q.shape[0]:
            o_padded = q.new_empty((q.shape[0], o.shape[1]), dtype=o.dtype)
            o_padded[: o.shape[0]] = o
            o = o_padded

        # 5) Output convention (same as before):
        #    Under the piecewise-CUDA-graph path, attention is invoked through the
        #    `unified_attention_with_output` split op which pre-allocates
        #    `forward_batch._attn_output`. In that path the op's return value is
        #    discarded, so we MUST write the result in place. When the backend is
        #    called directly (no forward context / non-piecewise), `_attn_output`
        #    is absent and the returned tensor is used instead.
        attn_output = getattr(forward_batch, "_attn_output", None)
        if attn_output is not None:
            attn_output.copy_(o.view_as(attn_output))
            return attn_output
        return o

    # ------------------------------------------------------------------ #
    # Forward: decode (delegated to standard FA3, NOT block-sparse)        #
    # ------------------------------------------------------------------ #
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Block selection is pointless for a single query token; use full
        # paged attention via the standard backend.
        return self.decode_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    # ------------------------------------------------------------------ #
    # CUDA graph: only decode needs it, and decode == FA3 -> delegate.     #
    # ------------------------------------------------------------------ #
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.decode_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
    ):
        self.decode_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self.decode_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.decode_backend.get_cuda_graph_seq_len_fill_value()
