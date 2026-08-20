#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <cub/device/device_scan.cuh>
#include "flash_block_sparse_index_wgmma.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

struct WorkspaceLayout {
    size_t k_mean_offset;
    size_t log_mass_offset;
    size_t counts_offset;
    size_t scan_offset;
    size_t total_bytes;
};

inline size_t align_up(size_t value, size_t alignment = 256) {
    return (value + alignment - 1) / alignment * alignment;
}

inline WorkspaceLayout make_workspace_layout(
        int batch_size, int num_kv_heads, int total_q_tiles,
        int max_k_blocks, int head_dim, bool is_fp8) {
    const size_t mean_elements = static_cast<size_t>(batch_size) * max_k_blocks
        * num_kv_heads * head_dim;
    const size_t score_elements = static_cast<size_t>(num_kv_heads)
        * total_q_tiles * max_k_blocks;
    const size_t segments = static_cast<size_t>(num_kv_heads) * total_q_tiles;
    size_t scan_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, scan_bytes, static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr), segments);

    WorkspaceLayout layout{};
    layout.k_mean_offset = 0;
    layout.log_mass_offset = align_up(mean_elements * (is_fp8 ? 1 : 2));
    layout.counts_offset = align_up(layout.log_mass_offset + score_elements * sizeof(float));
    layout.scan_offset = align_up(layout.counts_offset + segments * sizeof(int));
    layout.total_bytes = align_up(layout.scan_offset + scan_bytes);
    return layout;
}

template <typename T>
__device__ __forceinline__ float load_float(const T* ptr) {
    return static_cast<float>(*ptr);
}

template <typename T>
__device__ __forceinline__ T cast_input(float value) {
    return T(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_input(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ void online_add(float value, float& maximum, float& sum) {
    if (value > maximum) {
        sum = maximum == -CUDART_INF_F ? 1.0f : sum * __expf(maximum - value) + 1.0f;
        maximum = value;
    } else if (value != -CUDART_INF_F) {
        sum += __expf(value - maximum);
    }
}

__device__ __forceinline__ void merge_softmax(
        float other_max, float other_sum, float& maximum, float& sum) {
    if (other_max == -CUDART_INF_F) {
        return;
    }
    if (maximum == -CUDART_INF_F) {
        maximum = other_max;
        sum = other_sum;
    } else if (other_max > maximum) {
        sum = other_sum + sum * __expf(maximum - other_max);
        maximum = other_max;
    } else {
        sum += other_sum * __expf(other_max - maximum);
    }
}

template <typename T>
__global__ void paged_k_mean_kernel(
        const T* __restrict__ k_cache,
        T* __restrict__ k_mean,
        const int* __restrict__ page_table,
        const int* __restrict__ kv_seqlens,
        int batch_size, int num_kv_heads, int max_k_blocks,
        int page_size, int k_block_n, int head_dim,
        int64_t k_stride_page, int64_t k_stride_token,
        int64_t k_stride_head, int64_t page_table_stride_batch) {
    const int k_block = blockIdx.x;
    const int batch_head = blockIdx.y;
    const int batch = batch_head / num_kv_heads;
    const int kv_head = batch_head % num_kv_heads;
    if (batch >= batch_size || k_block >= max_k_blocks) {
        return;
    }

    const int kv_len = kv_seqlens[batch];
    const int token_begin = k_block * k_block_n;
    const int token_end = min(token_begin + k_block_n, kv_len);
    const int count = max(token_end - token_begin, 1);
    const size_t mean_base =
        ((static_cast<size_t>(batch) * max_k_blocks + k_block) * num_kv_heads
         + kv_head) * head_dim;

    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (int token = token_begin; token < token_end; ++token) {
            const int logical_page = token / page_size;
            const int page_offset = token - logical_page * page_size;
            const int physical_page = page_table[
                static_cast<int64_t>(batch) * page_table_stride_batch + logical_page];
            const int64_t offset = static_cast<int64_t>(physical_page) * k_stride_page
                + static_cast<int64_t>(page_offset) * k_stride_token
                + static_cast<int64_t>(kv_head) * k_stride_head + dim;
            sum += load_float(k_cache + offset);
        }
        k_mean[mean_base + dim] = cast_input<T>(sum / count);
    }
}

__device__ __forceinline__ int find_batch(
        int global_q_tile, const int* __restrict__ cu_q_tiles, int batch_size) {
    int low = 0;
    int high = batch_size;
    while (low + 1 < high) {
        const int mid = (low + high) >> 1;
        if (cu_q_tiles[mid] <= global_q_tile) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}

template <typename T>
__global__ void packgqa_log_mass_scalar_kernel(
        const T* __restrict__ q,
        const T* __restrict__ k_mean,
        float* __restrict__ log_mass,
        const int* __restrict__ cu_seqlens_q,
        const int* __restrict__ kv_seqlens,
        const int* __restrict__ cu_q_tiles,
        int batch_size, int num_q_heads, int num_kv_heads,
        int total_q_tiles, int max_k_blocks, int head_dim,
        int k_block_m, int k_block_n, float scale, bool causal,
        int64_t q_stride_token, int64_t q_stride_head) {
    const int global_q_tile = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (global_q_tile >= total_q_tiles || kv_head >= num_kv_heads) {
        return;
    }

    __shared__ float reduction_max[kThreads];
    __shared__ float reduction_sum[kThreads];
    const int batch = find_batch(global_q_tile, cu_q_tiles, batch_size);
    const int m_block = global_q_tile - cu_q_tiles[batch];
    const int q_begin = cu_seqlens_q[batch];
    const int q_len = cu_seqlens_q[batch + 1] - q_begin;
    const int kv_len = kv_seqlens[batch];
    const int prefix_len = kv_len - q_len;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int packed_begin = m_block * k_block_m;
    const int packed_end = min(packed_begin + k_block_m, q_len * gqa_ratio);

    for (int k_block = 0; k_block < max_k_blocks; ++k_block) {
        float thread_max = -CUDART_INF_F;
        float thread_sum = 0.0f;
        const int k_last_token = (k_block + 1) * k_block_n - 1;
        const bool valid_k = k_block * k_block_n < kv_len;
        const size_t mean_base =
            ((static_cast<size_t>(batch) * max_k_blocks + k_block) * num_kv_heads
             + kv_head) * head_dim;

        if (valid_k) {
            for (int packed = packed_begin + threadIdx.x;
                 packed < packed_end; packed += blockDim.x) {
                const int q_pos = packed / gqa_ratio;
                if (!causal || prefix_len + q_pos >= k_last_token) {
                    const int q_head = kv_head * gqa_ratio + packed % gqa_ratio;
                    const int64_t q_base = static_cast<int64_t>(q_begin + q_pos)
                        * q_stride_token + static_cast<int64_t>(q_head) * q_stride_head;
                    float dot = 0.0f;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        dot += load_float(q + q_base + dim)
                            * load_float(k_mean + mean_base + dim);
                    }
                    online_add(dot * scale, thread_max, thread_sum);
                }
            }
        }

        reduction_max[threadIdx.x] = thread_max;
        reduction_sum[threadIdx.x] = thread_sum;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                merge_softmax(
                    reduction_max[threadIdx.x + offset],
                    reduction_sum[threadIdx.x + offset],
                    reduction_max[threadIdx.x], reduction_sum[threadIdx.x]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const size_t output =
                (static_cast<size_t>(kv_head) * total_q_tiles + global_q_tile)
                * max_k_blocks + k_block;
            log_mass[output] = reduction_max[0] == -CUDART_INF_F
                ? -CUDART_INF_F : reduction_max[0] + __logf(reduction_sum[0]);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void packgqa_log_mass_tensorcore_128_kernel(
        const T* __restrict__ q,
        const T* __restrict__ k_mean,
        float* __restrict__ log_mass,
        const int* __restrict__ cu_seqlens_q,
        const int* __restrict__ kv_seqlens,
        const int* __restrict__ cu_q_tiles,
        int batch_size, int num_q_heads, int num_kv_heads,
        int total_q_tiles, int max_k_blocks, int k_block_n,
        float scale, bool causal, int64_t q_stride_token,
        int64_t q_stride_head) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    constexpr int kM = 128;
    constexpr int kN = 16;
    constexpr int kK = 128;
    __shared__ __nv_bfloat16 q_shared[kM * kK];
    __shared__ __nv_bfloat16 k_shared[kN * kK];
    __shared__ float score_shared[kM * kN];

    const int global_q_tile = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int batch = find_batch(global_q_tile, cu_q_tiles, batch_size);
    const int m_block = global_q_tile - cu_q_tiles[batch];
    const int q_begin = cu_seqlens_q[batch];
    const int q_len = cu_seqlens_q[batch + 1] - q_begin;
    const int kv_len = kv_seqlens[batch];
    const int prefix_len = kv_len - q_len;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int packed_begin = m_block * kM;
    const int packed_end = min(packed_begin + kM, q_len * gqa_ratio);

    for (int linear = threadIdx.x; linear < kM * kK; linear += blockDim.x) {
        const int row = linear / kK;
        const int dim = linear - row * kK;
        const int packed = packed_begin + row;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (packed < packed_end) {
            const int q_pos = packed / gqa_ratio;
            const int q_head = kv_head * gqa_ratio + packed % gqa_ratio;
            const int64_t offset = static_cast<int64_t>(q_begin + q_pos)
                * q_stride_token + static_cast<int64_t>(q_head) * q_stride_head + dim;
            value = __float2bfloat16_rn(load_float(q + offset));
        }
        q_shared[linear] = value;
    }
    __syncthreads();

    const int q_last = (packed_end - 1) / gqa_ratio;
    const int max_visible_token = prefix_len + q_last;
    for (int k_base = 0; k_base < max_k_blocks; k_base += kN) {
        if (causal && k_base * k_block_n + k_block_n - 1 > max_visible_token) {
            for (int k_block = k_base + threadIdx.x;
                 k_block < max_k_blocks; k_block += blockDim.x) {
                const size_t output =
                    (static_cast<size_t>(kv_head) * total_q_tiles + global_q_tile)
                    * max_k_blocks + k_block;
                log_mass[output] = -CUDART_INF_F;
            }
            return;
        }
        for (int linear = threadIdx.x; linear < kN * kK; linear += blockDim.x) {
            const int col = linear / kK;
            const int dim = linear - col * kK;
            const int k_block = k_base + col;
            __nv_bfloat16 value = __float2bfloat16(0.0f);
            if (k_block < max_k_blocks && k_block * k_block_n < kv_len) {
                const size_t offset =
                    ((static_cast<size_t>(batch) * max_k_blocks + k_block)
                     * num_kv_heads + kv_head) * kK + dim;
                value = __float2bfloat16_rn(load_float(k_mean + offset));
            }
            k_shared[linear] = value;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (int dim = 0; dim < kK; dim += 16) {
            wmma::load_matrix_sync(a_frag, q_shared + warp * 16 * kK + dim, kK);
            wmma::load_matrix_sync(b_frag, k_shared + dim, kK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(score_shared + warp * 16 * kN, c_frag, kN,
                                wmma::mem_row_major);
        __syncthreads();

        const int lane = threadIdx.x & 31;
        for (int col = warp; col < kN; col += 8) {
            const int k_block = k_base + col;
            const bool valid_k = k_block < max_k_blocks
                && k_block * k_block_n < kv_len;
            const int k_last_token = (k_block + 1) * k_block_n - 1;
            float maximum = -CUDART_INF_F;
            if (valid_k) {
                for (int row = lane; row < kM; row += 32) {
                    const int packed = packed_begin + row;
                    if (packed < packed_end) {
                        const int q_pos = packed / gqa_ratio;
                        if (!causal || prefix_len + q_pos >= k_last_token) {
                            maximum = fmaxf(maximum,
                                score_shared[row * kN + col] * scale);
                        }
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                maximum = fmaxf(maximum,
                    __shfl_down_sync(0xffffffff, maximum, offset));
            }
            maximum = __shfl_sync(0xffffffff, maximum, 0);
            float sum = 0.0f;
            if (valid_k && maximum != -CUDART_INF_F) {
                for (int row = lane; row < kM; row += 32) {
                    const int packed = packed_begin + row;
                    if (packed < packed_end) {
                        const int q_pos = packed / gqa_ratio;
                        if (!causal || prefix_len + q_pos >= k_last_token) {
                            const float value = score_shared[row * kN + col] * scale;
                            sum += exp2f((value - maximum) * 1.4426950408889634f);
                        }
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0 && k_block < max_k_blocks) {
                const size_t output =
                    (static_cast<size_t>(kv_head) * total_q_tiles + global_q_tile)
                    * max_k_blocks + k_block;
                log_mass[output] = maximum == -CUDART_INF_F
                    ? -CUDART_INF_F : maximum + __logf(sum);
            }
        }
        __syncthreads();
    }
#endif
}

struct SegmentInfo {
    int kv_len;
    int first_k;
    int last_k;
    bool full_last_tile;
};

__device__ __forceinline__ SegmentInfo get_segment_info(
        int global_q_tile, const int* cu_seqlens_q, const int* kv_seqlens,
        const int* cu_q_tiles, int batch_size, int gqa_ratio,
        int k_block_m, int k_block_n, int last_n_blocks) {
    const int batch = find_batch(global_q_tile, cu_q_tiles, batch_size);
    const int m_block = global_q_tile - cu_q_tiles[batch];
    const int q_tile_count = cu_q_tiles[batch + 1] - cu_q_tiles[batch];
    const int q_len = cu_seqlens_q[batch + 1] - cu_seqlens_q[batch];
    const int kv_len = kv_seqlens[batch];
    const int prefix_len = kv_len - q_len;
    const int packed_begin = m_block * k_block_m;
    const int packed_end = min(packed_begin + k_block_m, q_len * gqa_ratio);
    const int q_first = packed_begin / gqa_ratio;
    const int q_last = (packed_end - 1) / gqa_ratio;
    return {kv_len, (prefix_len + q_first) / k_block_n,
            (prefix_len + q_last) / k_block_n,
            m_block >= max(q_tile_count - last_n_blocks, 0)};
}

__device__ __forceinline__ bool keep_block(
        int k_block, float score, float max_score, const SegmentInfo& info,
        int k_block_n, float log_alpha, int attention_sink, int window_size,
        bool causal) {
    bool valid = k_block * k_block_n < info.kv_len;
    if (causal) {
        valid = valid && k_block <= info.last_k;
    }
    const bool scored = score != -CUDART_INF_F && score >= max_score + log_alpha;
    const bool sink = k_block < attention_sink;
    const bool local = k_block >= info.first_k - window_size + 1
        && k_block <= info.last_k;
    return valid && (scored || sink || local || info.full_last_tile);
}

__global__ void count_sparse_blocks_kernel(
        const float* __restrict__ log_mass, int* __restrict__ counts,
        const int* __restrict__ cu_seqlens_q,
        const int* __restrict__ kv_seqlens,
        const int* __restrict__ cu_q_tiles,
        int batch_size, int num_kv_heads, int total_q_tiles, int max_k_blocks,
        int gqa_ratio, int k_block_m, int k_block_n, float log_alpha,
        int attention_sink, int window_size, int last_n_blocks, bool causal) {
    const int segment = blockIdx.x;
    const int global_q_tile = segment % total_q_tiles;
    __shared__ float warp_max[kWarps];
    __shared__ int warp_count[kWarps];

    float local_max = -CUDART_INF_F;
    const float* scores = log_mass + static_cast<size_t>(segment) * max_k_blocks;
    for (int k = threadIdx.x; k < max_k_blocks; k += blockDim.x) {
        local_max = fmaxf(local_max, scores[k]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if ((threadIdx.x & 31) == 0) {
        warp_max[threadIdx.x >> 5] = local_max;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        local_max = threadIdx.x < kWarps ? warp_max[threadIdx.x] : -CUDART_INF_F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        if (threadIdx.x == 0) {
            warp_max[0] = local_max;
        }
    }
    __syncthreads();

    const SegmentInfo info = get_segment_info(
        global_q_tile, cu_seqlens_q, kv_seqlens, cu_q_tiles, batch_size,
        gqa_ratio, k_block_m, k_block_n, last_n_blocks);
    int local_count = 0;
    for (int k = threadIdx.x; k < max_k_blocks; k += blockDim.x) {
        local_count += keep_block(k, scores[k], warp_max[0], info, k_block_n,
                                  log_alpha, attention_sink, window_size, causal);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        warp_count[threadIdx.x >> 5] = local_count;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        int total = threadIdx.x < kWarps ? warp_count[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (threadIdx.x == 0) {
            counts[segment] = total;
        }
    }
}

__global__ void fill_sparse_blocks_kernel(
        const float* __restrict__ log_mass,
        const int* __restrict__ block_sparse_cu,
        int* __restrict__ block_sparse_idx,
        const int* __restrict__ cu_seqlens_q,
        const int* __restrict__ kv_seqlens,
        const int* __restrict__ cu_q_tiles,
        int batch_size, int total_q_tiles, int max_k_blocks, int gqa_ratio,
        int k_block_m, int k_block_n, float log_alpha, int attention_sink,
        int window_size, int last_n_blocks, bool causal) {
    const int segment = blockIdx.x;
    const int global_q_tile = segment % total_q_tiles;
    const float* scores = log_mass + static_cast<size_t>(segment) * max_k_blocks;
    __shared__ float shared_max;
    __shared__ int warp_counts[kWarps];
    __shared__ int warp_offsets[kWarps];
    __shared__ int chunk_base;

    float local_max = -CUDART_INF_F;
    for (int k = threadIdx.x; k < max_k_blocks; k += blockDim.x) {
        local_max = fmaxf(local_max, scores[k]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if ((threadIdx.x & 31) == 0) {
        reinterpret_cast<float*>(warp_offsets)[threadIdx.x >> 5] = local_max;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = -CUDART_INF_F;
        for (int warp = 0; warp < kWarps; ++warp) {
            maximum = fmaxf(maximum, reinterpret_cast<float*>(warp_offsets)[warp]);
        }
        shared_max = maximum;
        chunk_base = block_sparse_cu[segment];
    }
    __syncthreads();

    const SegmentInfo info = get_segment_info(
        global_q_tile, cu_seqlens_q, kv_seqlens, cu_q_tiles, batch_size,
        gqa_ratio, k_block_m, k_block_n, last_n_blocks);
    for (int chunk = 0; chunk < max_k_blocks; chunk += blockDim.x) {
        const int k = chunk + threadIdx.x;
        const bool keep = k < max_k_blocks
            && keep_block(k, scores[k], shared_max, info, k_block_n, log_alpha,
                          attention_sink, window_size, causal);
        const unsigned mask = __ballot_sync(0xffffffff, keep);
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int lane_rank = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0) {
            warp_counts[warp] = __popc(mask);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int prefix = 0;
            for (int w = 0; w < kWarps; ++w) {
                warp_offsets[w] = prefix;
                prefix += warp_counts[w];
            }
        }
        __syncthreads();
        if (keep) {
            block_sparse_idx[chunk_base + warp_offsets[warp] + lane_rank] = k;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int chunk_count = 0;
            for (int w = 0; w < kWarps; ++w) {
                chunk_count += warp_counts[w];
            }
            chunk_base += chunk_count;
        }
        __syncthreads();
    }
}

template <typename T>
cudaError_t launch_impl(
        const void* q, const void* k_cache, const int* page_table,
        const int* cu_seqlens_q, const int* kv_seqlens, const int* cu_q_tiles,
        int* block_sparse_cu, int* block_sparse_idx, void* workspace,
        size_t workspace_bytes, int batch_size, int num_q_heads,
        int num_kv_heads, int total_q_tiles, int max_k_blocks, int head_dim,
        int page_size, int k_block_m, int k_block_n, float alpha, float scale,
        int attention_sink, int window_size, int last_n_blocks, bool causal,
        int64_t q_stride_token, int64_t q_stride_head,
        int64_t k_stride_page, int64_t k_stride_token, int64_t k_stride_head,
        int64_t page_table_stride_batch, cudaStream_t stream, bool is_fp8) {
    const WorkspaceLayout layout = make_workspace_layout(
        batch_size, num_kv_heads, total_q_tiles, max_k_blocks, head_dim, is_fp8);
    if (workspace_bytes < layout.total_bytes) {
        return cudaErrorInvalidValue;
    }
    auto* bytes = static_cast<unsigned char*>(workspace);
    auto* k_mean = reinterpret_cast<T*>(bytes + layout.k_mean_offset);
    auto* log_mass = reinterpret_cast<float*>(bytes + layout.log_mass_offset);
    auto* counts = reinterpret_cast<int*>(bytes + layout.counts_offset);
    void* scan_workspace = bytes + layout.scan_offset;
    size_t scan_bytes = layout.total_bytes - layout.scan_offset;

    const dim3 mean_grid(max_k_blocks, batch_size * num_kv_heads);
    paged_k_mean_kernel<T><<<mean_grid, kThreads, 0, stream>>>(
        static_cast<const T*>(k_cache), k_mean, page_table, kv_seqlens,
        batch_size, num_kv_heads, max_k_blocks, page_size, k_block_n, head_dim,
        k_stride_page, k_stride_token, k_stride_head, page_table_stride_batch);
    const dim3 score_grid(total_q_tiles, num_kv_heads);
    if (head_dim == 128 && k_block_m == 128) {
        sparse_wgmma::packgqa_log_mass_wgmma_128_kernel<T><<<score_grid, 128, 0, stream>>>(
            static_cast<const T*>(q), k_mean, log_mass, cu_seqlens_q, kv_seqlens,
            cu_q_tiles, batch_size, num_q_heads, num_kv_heads, total_q_tiles,
            max_k_blocks, k_block_n, scale, causal, q_stride_token, q_stride_head);
    } else {
        packgqa_log_mass_scalar_kernel<T><<<score_grid, kThreads, 0, stream>>>(
            static_cast<const T*>(q), k_mean, log_mass, cu_seqlens_q, kv_seqlens,
            cu_q_tiles, batch_size, num_q_heads, num_kv_heads, total_q_tiles,
            max_k_blocks, head_dim, k_block_m, k_block_n, scale, causal,
            q_stride_token, q_stride_head);
    }

    const int segments = num_kv_heads * total_q_tiles;
    const float log_alpha = alpha == 0.0f ? -std::numeric_limits<float>::infinity() : logf(alpha);
    count_sparse_blocks_kernel<<<segments, kThreads, 0, stream>>>(
        log_mass, counts, cu_seqlens_q, kv_seqlens, cu_q_tiles, batch_size,
        num_kv_heads, total_q_tiles, max_k_blocks, num_q_heads / num_kv_heads,
        k_block_m, k_block_n, log_alpha, attention_sink, window_size,
        last_n_blocks, causal);
    cudaMemsetAsync(block_sparse_cu, 0, sizeof(int), stream);
    cub::DeviceScan::InclusiveSum(
        scan_workspace, scan_bytes, counts, block_sparse_cu + 1, segments, stream);
    fill_sparse_blocks_kernel<<<segments, kThreads, 0, stream>>>(
        log_mass, block_sparse_cu, block_sparse_idx, cu_seqlens_q, kv_seqlens,
        cu_q_tiles, batch_size, total_q_tiles, max_k_blocks,
        num_q_heads / num_kv_heads, k_block_m, k_block_n, log_alpha,
        attention_sink, window_size, last_n_blocks, causal);
    return cudaGetLastError();
}

}  // namespace

extern "C" size_t flash_block_sparse_index_workspace_size_sm90(
        int batch_size, int num_kv_heads, int total_q_tiles,
        int max_k_blocks, int head_dim, bool is_fp8) {
    return make_workspace_layout(batch_size, num_kv_heads, total_q_tiles,
                                 max_k_blocks, head_dim, is_fp8).total_bytes;
}

extern "C" cudaError_t build_block_sparse_index_sm90(
        const void* q, const void* k_cache, const int* page_table,
        const int* cu_seqlens_q, const int* kv_seqlens, const int* cu_q_tiles,
        int* block_sparse_cu, int* block_sparse_idx, void* workspace,
        size_t workspace_bytes, int batch_size, int num_q_heads,
        int num_kv_heads, int total_q_tiles, int max_k_blocks, int head_dim,
        int page_size, int k_block_m, int k_block_n, float alpha,
        int attention_sink, int window_size, int last_n_blocks, bool causal,
        float softmax_scale, float q_descale, float k_descale,
        int64_t q_stride_token, int64_t q_stride_head,
        int64_t k_stride_page, int64_t k_stride_token, int64_t k_stride_head,
        int64_t page_table_stride_batch, bool is_fp8, void* stream_void) {
    if (!q || !k_cache || !page_table || !cu_seqlens_q || !kv_seqlens
        || !cu_q_tiles || !block_sparse_cu || !block_sparse_idx || !workspace
        || batch_size < 0 || num_q_heads <= 0 || num_kv_heads <= 0
        || num_q_heads % num_kv_heads != 0 || total_q_tiles <= 0
        || max_k_blocks <= 0 || head_dim <= 0 || page_size <= 0
        || k_block_m <= 0 || k_block_n <= 0 || alpha < 0.0f || alpha > 1.0f
        || attention_sink < 0 || window_size < 0 || last_n_blocks < 0) {
        return cudaErrorInvalidValue;
    }
    const float scale = softmax_scale * q_descale * k_descale;
    const cudaStream_t stream = static_cast<cudaStream_t>(stream_void);
    if (is_fp8) {
        return launch_impl<__nv_fp8_e4m3>(
            q, k_cache, page_table, cu_seqlens_q, kv_seqlens, cu_q_tiles,
            block_sparse_cu, block_sparse_idx, workspace, workspace_bytes,
            batch_size, num_q_heads, num_kv_heads, total_q_tiles, max_k_blocks,
            head_dim, page_size, k_block_m, k_block_n, alpha, scale,
            attention_sink, window_size, last_n_blocks, causal,
            q_stride_token, q_stride_head, k_stride_page, k_stride_token,
            k_stride_head, page_table_stride_batch, stream, true);
    }
    return launch_impl<__nv_bfloat16>(
        q, k_cache, page_table, cu_seqlens_q, kv_seqlens, cu_q_tiles,
        block_sparse_cu, block_sparse_idx, workspace, workspace_bytes,
        batch_size, num_q_heads, num_kv_heads, total_q_tiles, max_k_blocks,
        head_dim, page_size, k_block_m, k_block_n, alpha, scale,
        attention_sink, window_size, last_n_blocks, causal,
        q_stride_token, q_stride_head, k_stride_page, k_stride_token,
        k_stride_head, page_table_stride_batch, stream, false);
}