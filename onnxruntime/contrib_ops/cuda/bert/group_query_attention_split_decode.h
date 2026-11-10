// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "core/common/status.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Single-token decode attention for GroupQueryAttention on a BNSH KV cache (fp16, or 8-bit INT8 / FP8 E4M3
// with PER_TENSOR or PER_CHANNEL scales) that already holds the new token (LaunchGqaSplitDecodePrep has
// appended it).
//
// An 8-bit cache is read as is (half the bytes of fp16) and widened to fp16 in registers, exactly: INT8 by
// the 0x6400 magic-number trick, E4M3 by moving its bits into a half (2^-8 of the value, folded into the
// scales). A PER_TENSOR K scale is folded into the softmax scale; a PER_CHANNEL one multiplies the widened
// K fragments in fp32 before rounding to fp16 (the value DequantizeKVPair produces); V scales multiply the
// output channels at the end. Nothing is dequantized in memory.
//
// The flash path for this case launches a split-KV kernel over all query heads, a combine kernel and
// two memsets of the split accumulators. This kernel runs one CTA per (batch, kv head, split); its warps
// stream 8-key steps of K and V through a cp.async ring in shared memory into tensor-core mma fragments,
// with the 8 query heads of the KV head as the rows of one m16 tile, so K and V are read once for the
// whole group. The warps of a CTA are combined in shared memory and the last CTA of a (batch, kv head)
// merges the per-split softmax partials, so there is no second kernel and no memset.
// Requires sm_80 (the kernel body is empty on older architectures; the Flash path gates it).
//
// A CUDA graph freezes the grid, so the kernel is launched with GqaSplitDecodeMaxSplits() splits per
// (batch, kv head), which depends only on the batch size, the KV head count, the KV buffer length and
// the GPU (one CTA per SM at a time). Each batch then uses
// ceil(total_seq_len / kGqaSplitDecodeMinKeysPerSplit) of them (capped by the launched count), read
// from total_seq_lens on the device; the remaining CTAs exit at once.
constexpr int kGqaSplitDecodeMinKeysPerSplit = 128;
constexpr int kGqaSplitDecodeMaxSplits = 64;
constexpr int kGqaSplitDecodeGroup = 8;
constexpr int kGqaSplitDecodeHeadSize = 128;

// Scratch for one run, sized for `batch_size` x `kv_num_heads` (x `max_splits` for the partials):
//   partials: per-split softmax partials (contents need no initialization)
//   counters: one per (batch, kv head); LaunchGqaSplitDecodePrep zeroes them for the attention kernel.
int GqaSplitDecodeMaxSplits(int batch_size, int kv_num_heads, int capacity, int num_sms);
size_t GqaSplitDecodePartialsBytes(int batch_size, int kv_num_heads, int max_splits);
size_t GqaSplitDecodeCountersBytes(int batch_size, int kv_num_heads);

// Element type of the KV cache. The 8-bit caches hold symmetric INT8 or FP8 E4M3 values of x / scale,
// quantized exactly as UnpackRoPEAppend does (round to nearest and clamp to [-128, 127], or a saturating
// E4M3 conversion of x / scale clamped to +-448), so a cache written by the prompt path and one appended
// to here agree element for element.
enum class GqaSplitDecodeCache { kFp16, kInt8, kFp8E4M3 };

// Cache element type and dequantization scales (fp32 on the device). A scale is one value (PER_TENSOR)
// or kv_num_heads * head_size values (PER_CHANNEL, the (kv_num_heads, 1, head_size) tensor flattened).
// Unused for kFp16.
struct GqaSplitDecodeQuant {
  GqaSplitDecodeCache cache = GqaSplitDecodeCache::kFp16;
  const float* k_scale = nullptr;
  const float* v_scale = nullptr;
  bool k_per_channel = false;
  bool v_per_channel = false;
};

// A paged cache (PagedAttention): key/value caches [num_blocks, block_size, kv_num_heads, head_size], and
// sequence b's position t at block block_table[b][t / block_size], slot t % block_size. block_size must be a
// multiple of 8 (one 8-key step never crosses a block). With `paged` null the cache is the BNSH
// [batch, kv_num_heads, capacity, head_size] buffer of GroupQueryAttention.
//
// For a paged cache the prep kernel also derives the per-sequence lengths the attention kernel reads:
// sequence b is valid when it has exactly one query token (cumulative_seqlens_q[b + 1] - [b] == 1, token b)
// and, with `sequence_validity`, is marked valid; it then appends at past_seqlens[b] (or slot_mapping[b])
// and total_seq_lens[b] = past + 1. An invalid sequence gets total 0: no append and a zero output.
struct GqaSplitDecodePaged {
  const int* block_table = nullptr;           // [batch, max_blocks_per_seq]
  const int* slot_mapping = nullptr;          // [batch] flat slot (block * block_size + offset), nullable
  const int* cumulative_seqlens_q = nullptr;  // [batch + 1]
  const int* sequence_validity = nullptr;     // [batch] nonzero = valid, nullable (all valid)
  int* total_seq_lens = nullptr;              // [batch] written by the prep kernel
  int block_size = 0;
  int max_blocks_per_seq = 0;
};

// Decode-step preparation for the split decode path, replacing UnpackRoPEAppend for one token:
// per (head, batch) one warp loads the head, applies QK-Norm (fp32 statistics, result rounded to
// fp16, as UnpackRoPEAppend does), half-split RoPE over the full head, then writes Q to q_out and
// K/V into the BNSH cache at past_seq_lens[b] (quantized for an 8-bit cache). Inputs are either packed_qkv
// [batch, 1, (num_heads + 2 * kv_num_heads) * head_size] or separate q/k/v (packed_qkv == nullptr).
// position_ids [batch] (nullable: past_seq_lens[b] is the position); cos/sin [rope_max_pos, head_size / 2]
// (nullable when do_rotary is false); q_norm/k_norm [head_size] (nullable). It also zeroes `counters`
// [batch, kv_num_heads] for the LaunchGqaSplitDecode call that follows on the same stream.
// k_cache / v_cache hold half, int8_t or __nv_fp8_e4m3 elements as quant.cache says. For a paged cache
// `capacity` is max_blocks_per_seq * block_size and position_ids must be null.
Status LaunchGqaSplitDecodePrep(cudaStream_t stream, const half* packed_qkv, const half* q, const half* k,
                                const half* v, half* q_out, void* k_cache, void* v_cache, const int* past_seq_lens,
                                const int64_t* position_ids, const half* cos_cache, const half* sin_cache,
                                int rope_max_pos, bool do_rotary, const half* q_norm, const half* k_norm, float q_eps,
                                float k_eps, unsigned int* counters, int batch_size, int num_heads, int kv_num_heads,
                                int capacity, const GqaSplitDecodeQuant& quant,
                                const GqaSplitDecodePaged* paged = nullptr);

// q:       [batch, 1, num_heads, head_size] (BSNH, rotated/normalized)
// k_cache: [batch, kv_num_heads, capacity, head_size] (BNSH), v_cache likewise, elements as quant.cache says;
//          or the paged layout above
// output:  [batch, 1, num_heads * head_size]
// total_seq_lens: [batch] keys per sequence (for a paged cache, paged->total_seq_lens as written by the prep)
Status LaunchGqaSplitDecode(cudaStream_t stream, const half* q, const void* k_cache, const void* v_cache,
                            half* output, const int* total_seq_lens, float scale, int batch_size, int kv_num_heads,
                            int capacity, int max_splits, float* partials, unsigned int* counters,
                            const GqaSplitDecodeQuant& quant, const GqaSplitDecodePaged* paged = nullptr);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
