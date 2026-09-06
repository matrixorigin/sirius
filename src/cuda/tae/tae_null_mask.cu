/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cudf/cudf_compat.hpp"

#include <cuda/tae/tae_decode_kernels.hpp>
#include <cuda_runtime.h>

#include <cstdint>

namespace sirius::cuda::tae {

namespace {

constexpr uint32_t THREADS_PER_BLOCK = 256;

// MO null bitmap: bit=1 → NULL.
// cuDF validity mask: bit=1 → VALID.
// Inversion: bitwise NOT each 32-bit word.
//
// An MO vector has a 29-byte header followed by variable-sized sections, so
// the nsp bitmap is not guaranteed to be naturally aligned. Keep the source
// byte-addressed: casting it to uint32_t/uint4_t faults on GPUs that enforce
// alignment. The cuDF output mask is allocator-aligned and remains word-based.
__device__ uint32_t load_unaligned_le_u32(const uint8_t* src)
{
  return static_cast<uint32_t>(src[0]) | (static_cast<uint32_t>(src[1]) << 8) |
         (static_cast<uint32_t>(src[2]) << 16) | (static_cast<uint32_t>(src[3]) << 24);
}

__global__ void invert_mask_kernel(const uint8_t* __restrict__ src,
                                   uint32_t* __restrict__ dst,
                                   uint32_t n_words)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words;
       i += gridDim.x * blockDim.x) {
    dst[i] = ~load_unaligned_le_u32(src + i * sizeof(uint32_t));
  }
}

// Batched null mask inversion — 2D grid: blockIdx.y = desc, blockIdx.x = word tile
__global__ void batched_invert_mask_kernel(const BatchedNullMaskDesc* __restrict__ descs,
                                           uint32_t* __restrict__ dst)
{
  auto const& desc = descs[blockIdx.y];
  uint32_t n_words = (desc.n_rows + 31) / 32;
  auto const* src  = desc.src;
  auto* out        = dst + desc.bitmask_word_offset;

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_words;
       i += gridDim.x * blockDim.x) {
    out[i] = ~load_unaligned_le_u32(src + i * sizeof(uint32_t));
  }
}

}  // anonymous namespace

void invert_null_mask(const uint8_t* d_src,
                      uint32_t* d_dst,
                      uint32_t n_rows,
                      rmm::cuda_stream_view stream)
{
  if (n_rows == 0) return;

  // Number of 32-bit words needed: ceil(n_rows / 32)
  uint32_t n_words = (n_rows + 31) / 32;
  uint32_t blocks  = (n_words + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  invert_mask_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(d_src, d_dst, n_words);
}

void batched_invert_null_mask(const BatchedNullMaskDesc* d_descs,
                              uint32_t n_descs,
                              uint32_t* d_validity,
                              rmm::cuda_stream_view stream)
{
  if (n_descs == 0) return;

  // Find max n_words across descriptors for grid X sizing.
  // Since descriptors are on device, use a conservative upper bound:
  // max 8192 rows → 256 words → 1 block at 256 threads.
  // Grid: (1, n_descs) is sufficient for typical TAE blocks.
  uint32_t max_words = (8192 + 31) / 32;  // 256 words
  uint32_t grid_x    = (max_words + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dim3 grid(grid_x, n_descs);
  batched_invert_mask_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.value()>>>(d_descs, d_validity);
}

}  // namespace sirius::cuda::tae
