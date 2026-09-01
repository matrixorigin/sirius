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

// sirius
#include <cuda/tae/tae_decode_kernels.hpp>

#include <data/host_tae_representation.hpp>
#include <data/host_tae_representation_converters.hpp>
#include <log/logging.hpp>
#include <tae/tae_format.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

// nvcomp
#include <nvcomp/lz4.h>

// rmm
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

// CUDA
#include <cuda_runtime.h>

// cudf error checking
#include <cudf/utilities/error.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sirius {

namespace detail {

// ---------------------------------------------------------------------------
// MO type → cuDF type mapping
// ---------------------------------------------------------------------------
static cudf::data_type mo_oid_to_cudf_type(tae::MOTypeOid oid, int32_t scale = 0)
{
  switch (oid) {
    case tae::MO_T_bool: return cudf::data_type{cudf::type_id::BOOL8};
    case tae::MO_T_int8: return cudf::data_type{cudf::type_id::INT8};
    case tae::MO_T_int16: return cudf::data_type{cudf::type_id::INT16};
    case tae::MO_T_int32: return cudf::data_type{cudf::type_id::INT32};
    case tae::MO_T_int64: return cudf::data_type{cudf::type_id::INT64};
    case tae::MO_T_uint8: return cudf::data_type{cudf::type_id::UINT8};
    case tae::MO_T_uint16: return cudf::data_type{cudf::type_id::UINT16};
    case tae::MO_T_uint32: return cudf::data_type{cudf::type_id::UINT32};
    case tae::MO_T_uint64: return cudf::data_type{cudf::type_id::UINT64};
    case tae::MO_T_float32: return cudf::data_type{cudf::type_id::FLOAT32};
    case tae::MO_T_float64: return cudf::data_type{cudf::type_id::FLOAT64};
    case tae::MO_T_date: return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    case tae::MO_T_datetime: return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
    case tae::MO_T_char:
    case tae::MO_T_varchar:
    case tae::MO_T_text:
    case tae::MO_T_blob:
    case tae::MO_T_json: return cudf::data_type{cudf::type_id::STRING};
    case tae::MO_T_decimal64: return cudf::data_type{cudf::type_id::DECIMAL64, -scale};
    case tae::MO_T_decimal128: return cudf::data_type{cudf::type_id::DECIMAL128, -scale};
    default: return cudf::data_type{cudf::type_id::INT64};
  }
}

// Fixed-width byte size for a MO type OID
static uint32_t mo_oid_fixed_width(tae::MOTypeOid oid)
{
  switch (oid) {
    case tae::MO_T_bool:
    case tae::MO_T_int8:
    case tae::MO_T_uint8: return 1;
    case tae::MO_T_int16:
    case tae::MO_T_uint16: return 2;
    case tae::MO_T_int32:
    case tae::MO_T_uint32:
    case tae::MO_T_float32:
    case tae::MO_T_date: return 4;
    case tae::MO_T_int64:
    case tae::MO_T_uint64:
    case tae::MO_T_float64:
    case tae::MO_T_datetime:
    case tae::MO_T_decimal64: return 8;
    case tae::MO_T_decimal128: return 16;
    default: return 0;  // variable-width
  }
}

static bool is_varchar_type(tae::MOTypeOid oid)
{
  switch (oid) {
    case tae::MO_T_char:
    case tae::MO_T_varchar:
    case tae::MO_T_text:
    case tae::MO_T_blob:
    case tae::MO_T_json: return true;
    default: return false;
  }
}

static bool is_date_type(tae::MOTypeOid oid) { return oid == tae::MO_T_date; }
static bool is_timestamp_type(tae::MOTypeOid oid) { return oid == tae::MO_T_datetime; }

// ---------------------------------------------------------------------------
// MO serialized vector header offsets.
// After LZ4 decompression, each column chunk is:
//   [IOEntryHeader 4B][class 1B][MOType 16B][row_count 4B][dataLen 4B]
//   [data dataLen B][areaLen 4B][area areaLen B]
//   [nspLen 4B][nsp nspLen B][sorted 1B]
//
// Data section starts at byte 29 from the decompressed start.
// For fixed-width columns: dataLen = row_count * elem_size, areaLen = 0
// For varchar columns:     dataLen = row_count * 24 (varlena structs), area = string payloads
// ---------------------------------------------------------------------------
constexpr uint32_t VARLENA_STRUCT_SIZE = 24;

// ---------------------------------------------------------------------------
// Convert host_tae_representation → gpu_table_representation
// ---------------------------------------------------------------------------
std::unique_ptr<cucascade::idata_representation> convert_host_tae_to_gpu(
  cucascade::idata_representation& source,
  cucascade::memory::memory_space const* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& host_src                         = source.cast<host_tae_representation>();
  auto const& chunks                     = host_src.get_column_chunks();
  auto const& post_filter_projection_ids = host_src.get_post_filter_projection_ids();

  if (chunks.empty()) {
    auto empty_table = std::make_unique<cudf::table>();
    auto result      = std::make_unique<cucascade::gpu_table_representation>(
      std::move(empty_table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
    host_src.mark_h2d_complete();
    return result;
  }

#ifdef SIRIUS_PROFILE
  auto const cvt_t0 = std::chrono::high_resolution_clock::now();
#endif

  rmm::device_async_resource_ref mr_ref(target_memory_space->get_default_allocator());
  rmm::cuda_device_id target_device_id(target_memory_space->get_device_id());
  rmm::cuda_set_device_raii target_device_raii(target_device_id);

  // 1. Get contiguous host buffer
  auto const& linear_host = *host_src.get_host_data();

  // 2. Group chunks by column_idx (ordered by block index within each column)
  //    Each column may have multiple blocks that need to be concatenated.
  struct chunk_ref {
    std::size_t chunk_index;
  };
  std::map<uint16_t, std::vector<chunk_ref>> col_chunks;
  for (std::size_t i = 0; i < chunks.size(); i++) {
    col_chunks[chunks[i].column_idx].push_back({i});
  }

  // 3. Separate compressed chunks and uncompressed chunks
  std::vector<std::size_t> compressed_indices;  // indices into chunks[]
  std::vector<std::size_t> uncompressed_indices;
  for (std::size_t i = 0; i < chunks.size(); i++) {
    if (chunks[i].extent.is_compressed()) {
      compressed_indices.push_back(i);
    } else {
      uncompressed_indices.push_back(i);
    }
  }

  // GPU-side events for non-blocking pipeline timing (queried after final sync)
#ifdef SIRIUS_PROFILE
  cudaEvent_t ev_pre_h2d, ev_h2d, ev_lz4, ev_decode, ev_filter, ev_apply, ev_end;
  CUDF_CUDA_TRY(cudaEventCreate(&ev_pre_h2d));
  CUDF_CUDA_TRY(cudaEventCreate(&ev_h2d));
  CUDF_CUDA_TRY(cudaEventCreate(&ev_lz4));
  CUDF_CUDA_TRY(cudaEventCreate(&ev_decode));
  CUDF_CUDA_TRY(cudaEventCreate(&ev_filter));
  CUDF_CUDA_TRY(cudaEventCreate(&ev_apply));
  CUDF_CUDA_TRY(cudaEventCreate(&ev_end));
#endif

  // 4. Transfer entire host buffer to GPU in a single contiguous copy.
  //    This replaces per-chunk H→D copies, reducing driver overhead and
  //    enabling full PCIe bandwidth utilization.
  rmm::device_buffer d_mirror(linear_host.size(), stream, mr_ref);
#ifdef SIRIUS_PROFILE
  CUDF_CUDA_TRY(cudaEventRecord(ev_pre_h2d, stream.value()));
#endif
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_mirror.data(),
                                linear_host.data(),
                                linear_host.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));
#ifdef SIRIUS_PROFILE
  auto const cvt_h2d = std::chrono::high_resolution_clock::now();
  CUDF_CUDA_TRY(cudaEventRecord(ev_h2d, stream.value()));
#endif

  // 5. Allocate decompression output buffers
  std::size_t total_compressed = 0;
  for (auto idx : compressed_indices) {
    total_compressed += chunks[idx].pinned_length;
  }

  std::size_t total_decompressed = 0;
  for (auto idx : compressed_indices) {
    total_decompressed += chunks[idx].extent.origin_size;
  }

  rmm::device_buffer d_decompressed(total_decompressed, stream, mr_ref);

  std::vector<std::size_t> d_decompressed_offsets(compressed_indices.size());
  {
    std::size_t off = 0;
    for (std::size_t ci = 0; ci < compressed_indices.size(); ci++) {
      d_decompressed_offsets[ci] = off;
      off += chunks[compressed_indices[ci]].extent.origin_size;
    }
  }

  // 6. nvCOMP batch LZ4 decompress (source pointers reference d_mirror directly)
  if (!compressed_indices.empty()) {
    std::size_t num_chunks_to_decompress = compressed_indices.size();

    // Build pointer and size arrays on device
    std::vector<void const*> h_comp_ptrs(num_chunks_to_decompress);
    std::vector<std::size_t> h_comp_sizes(num_chunks_to_decompress);
    std::vector<void*> h_decomp_ptrs(num_chunks_to_decompress);
    std::vector<std::size_t> h_decomp_buf_sizes(num_chunks_to_decompress);

    std::size_t max_uncomp_size = 0;

    for (std::size_t ci = 0; ci < num_chunks_to_decompress; ci++) {
      auto& chunk       = chunks[compressed_indices[ci]];
      h_comp_ptrs[ci]   = static_cast<uint8_t*>(d_mirror.data()) + chunk.pinned_offset;
      h_comp_sizes[ci]  = chunk.pinned_length;
      h_decomp_ptrs[ci] = static_cast<uint8_t*>(d_decompressed.data()) + d_decompressed_offsets[ci];
      h_decomp_buf_sizes[ci] = chunk.extent.origin_size;
      max_uncomp_size =
        std::max(max_uncomp_size, static_cast<std::size_t>(chunk.extent.origin_size));
    }

    // Device arrays
    rmm::device_uvector<void const*> d_comp_ptrs(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<std::size_t> d_comp_sizes(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<void*> d_decomp_ptrs(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<std::size_t> d_decomp_buf_sizes(num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<std::size_t> d_decomp_actual_sizes(
      num_chunks_to_decompress, stream, mr_ref);
    rmm::device_uvector<nvcompStatus_t> d_statuses(num_chunks_to_decompress, stream, mr_ref);

    CUDF_CUDA_TRY(cudaMemcpyAsync(d_comp_ptrs.data(),
                                  h_comp_ptrs.data(),
                                  num_chunks_to_decompress * sizeof(void*),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_comp_sizes.data(),
                                  h_comp_sizes.data(),
                                  num_chunks_to_decompress * sizeof(std::size_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_decomp_ptrs.data(),
                                  h_decomp_ptrs.data(),
                                  num_chunks_to_decompress * sizeof(void*),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_decomp_buf_sizes.data(),
                                  h_decomp_buf_sizes.data(),
                                  num_chunks_to_decompress * sizeof(std::size_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));

    // Get temp size
    std::size_t temp_bytes = 0;
    auto opts              = nvcompBatchedLZ4DecompressDefaultOpts;
    auto status            = nvcompBatchedLZ4DecompressGetTempSizeAsync(
      num_chunks_to_decompress, max_uncomp_size, opts, &temp_bytes, total_decompressed);
    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedLZ4DecompressGetTempSizeAsync failed: " +
                               std::to_string(status));
    }

    rmm::device_buffer d_temp(temp_bytes, stream, mr_ref);

    // Decompress
    status = nvcompBatchedLZ4DecompressAsync(d_comp_ptrs.data(),
                                             d_comp_sizes.data(),
                                             d_decomp_buf_sizes.data(),
                                             d_decomp_actual_sizes.data(),
                                             num_chunks_to_decompress,
                                             d_temp.data(),
                                             temp_bytes,
                                             d_decomp_ptrs.data(),
                                             opts,
                                             d_statuses.data(),
                                             stream.value());

    if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedLZ4DecompressAsync failed: " + std::to_string(status));
    }

#ifdef SIRIUS_PROFILE
    SIRIUS_LOG_INFO(
      "[tae_converter] LZ4 batch: {} chunks, compressed={:.2f}MB, decompressed={:.2f}MB, "
      "ratio={:.2f}x, max_chunk={:.1f}KB, min_chunk={:.1f}KB, temp={:.1f}KB",
      num_chunks_to_decompress,
      total_compressed / 1048576.0,
      total_decompressed / 1048576.0,
      static_cast<double>(total_decompressed) / std::max(total_compressed, (std::size_t)1),
      *std::max_element(h_decomp_buf_sizes.begin(), h_decomp_buf_sizes.end()) / 1024.0,
      *std::min_element(h_decomp_buf_sizes.begin(), h_decomp_buf_sizes.end()) / 1024.0,
      temp_bytes / 1024.0);
#endif
  }
#ifdef SIRIUS_PROFILE
  auto const cvt_lz4 = std::chrono::high_resolution_clock::now();
  CUDF_CUDA_TRY(cudaEventRecord(ev_lz4, stream.value()));
#endif

  // 7. Pre-built chunk→device_ptr map (O(1) lookup, replaces O(n) linear scan)
  std::vector<uint8_t*> chunk_device_ptrs(chunks.size());
  for (std::size_t ci = 0; ci < compressed_indices.size(); ci++) {
    chunk_device_ptrs[compressed_indices[ci]] =
      static_cast<uint8_t*>(d_decompressed.data()) + d_decompressed_offsets[ci];
  }
  for (auto idx : uncompressed_indices) {
    chunk_device_ptrs[idx] = static_cast<uint8_t*>(d_mirror.data()) + chunks[idx].pinned_offset;
  }

  auto get_chunk_decompressed_size = [&](std::size_t chunk_idx) -> std::size_t {
    auto& chunk = chunks[chunk_idx];
    return chunk.extent.is_compressed() ? chunk.extent.origin_size : chunk.pinned_length;
  };

  // 8. Decode columns using batched kernels (one launch per column instead of per-block).
  //    This reduces CUDA kernel launches from O(blocks × columns) to O(columns),
  //    eliminating ~10-15μs CPU-side overhead per launch × hundreds of launches.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(col_chunks.size());

  for (auto& [col_idx, chunk_refs] : col_chunks) {
    auto& first_chunk = chunks[chunk_refs[0].chunk_index];
    auto type_oid     = first_chunk.type_oid;
    auto cudf_type    = mo_oid_to_cudf_type(type_oid, first_chunk.scale);

    // Total rows for this column across all blocks
    std::size_t col_total_rows = 0;
    for (auto& cr : chunk_refs) {
      col_total_rows += chunks[cr.chunk_index].row_count;
    }

    if (is_varchar_type(type_oid)) {
      // VARCHAR: build cuDF offsets + chars from MO varlena format
      // MO vector layout: [header 29B][varlena_structs row_count*24B][areaLen 4B][area...][nsp...]

      // Build batched descriptors for all blocks of this column
      std::vector<cuda::tae::BatchedVarcharDesc> h_descs;
      h_descs.reserve(chunk_refs.size());
      uint32_t max_block_rows = 0;
      std::size_t row_offset  = 0;

      // Per-block metadata for null mask pass (host-side only)
      struct varchar_block_meta {
        uint32_t actual_data_len;
        uint32_t area_len;  // computed from extent metadata (no D2H needed)
        std::size_t chunk_index;
        uint32_t rows;
      };
      std::vector<varchar_block_meta> block_meta;
      block_meta.reserve(chunk_refs.size());
      std::size_t total_chars_ub = 0;  // upper bound for chars buffer (avoids D2H sync)

      for (std::size_t i = 0; i < chunk_refs.size(); i++) {
        auto& cr                 = chunk_refs[i];
        auto& chunk              = chunks[cr.chunk_index];
        auto* d_ptr              = chunk_device_ptrs[cr.chunk_index];
        auto d_size              = get_chunk_decompressed_size(cr.chunk_index);
        uint32_t actual_data_len = chunk.row_count * VARLENA_STRUCT_SIZE;

        if (chunk.vector_header_size + actual_data_len > d_size) {
          throw std::runtime_error("varchar varlena section exceeds decompressed buffer");
        }

        // Compute area_len from host extent metadata to avoid D2H + stream.synchronize().
        // Decompressed layout: [header 29B][data][areaLen 4B][area][nspLen 4B][nsp][sorted 1B]
        // => d_size = 38 + actual_data_len + area_len + nsp_len
        uint32_t nsp_len = 0;
        if (chunk.null_cnt > 0) {
          uint32_t nsp_words = (chunk.row_count + 63) / 64;
          nsp_len            = 24 + nsp_words * 8;
        }
        std::size_t fixed_overhead =
          chunk.vector_header_size + 9 + static_cast<std::size_t>(actual_data_len) + nsp_len;
        uint32_t area_len =
          (d_size > fixed_overhead) ? static_cast<uint32_t>(d_size - fixed_overhead) : 0;
        // Upper bound: each inline varlena can hold at most 23 chars + area holds big strings
        total_chars_ub += static_cast<std::size_t>(chunk.row_count) * 23 + area_len;

        h_descs.push_back({d_ptr + chunk.vector_header_size,
                           d_ptr + chunk.vector_header_size + actual_data_len + 4,
                           chunk.row_count,
                           static_cast<uint32_t>(row_offset)});
        block_meta.push_back({actual_data_len, area_len, cr.chunk_index, chunk.row_count});
        max_block_rows = std::max(max_block_rows, chunk.row_count);
        row_offset += chunk.row_count;
      }

      // Upload descriptors to device
      rmm::device_buffer d_descs(
        h_descs.size() * sizeof(cuda::tae::BatchedVarcharDesc), stream, mr_ref);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                                    h_descs.data(),
                                    h_descs.size() * sizeof(cuda::tae::BatchedVarcharDesc),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));

      // === Phase 1: batched compute_lengths + single global CUB ExclusiveSum ===
      auto offsets_buf = rmm::device_buffer((col_total_rows + 1) * sizeof(int32_t), stream, mr_ref);

      // CUB temp-size query
      std::size_t cub_temp_bytes = 0;
      cuda::tae::batched_decode_varchar_offsets(nullptr,
                                                static_cast<uint32_t>(h_descs.size()),
                                                nullptr,
                                                nullptr,
                                                cub_temp_bytes,
                                                static_cast<uint32_t>(col_total_rows),
                                                max_block_rows,
                                                stream);
      rmm::device_buffer d_cub_temp(cub_temp_bytes, stream, mr_ref);

      // Batched offsets: 1 kernel (compute_lengths) + 1 CUB (ExclusiveSum)
      std::size_t temp_bytes = cub_temp_bytes;
      cuda::tae::batched_decode_varchar_offsets(
        static_cast<cuda::tae::BatchedVarcharDesc*>(d_descs.data()),
        static_cast<uint32_t>(h_descs.size()),
        static_cast<int32_t*>(offsets_buf.data()),
        d_cub_temp.data(),
        temp_bytes,
        static_cast<uint32_t>(col_total_rows),
        max_block_rows,
        stream);

      // === Phase 2: batched scatter chars ===
      // Chars buffer sized from host-computed upper bound — no D2H or sync needed.
      auto chars_buf = rmm::device_buffer(total_chars_ub, stream, mr_ref);
      cuda::tae::batched_decode_varchar_scatter(
        static_cast<cuda::tae::BatchedVarcharDesc*>(d_descs.data()),
        static_cast<uint32_t>(h_descs.size()),
        static_cast<int32_t*>(offsets_buf.data()),
        static_cast<uint8_t*>(chars_buf.data()),
        max_block_rows,
        stream);

      // Build cuDF string column from offsets + chars
      auto offsets_col =
        std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(col_total_rows + 1),
                                       std::move(offsets_buf),
                                       rmm::device_buffer{},
                                       0);

      // Build null mask (batched across all blocks of this column)
      uint32_t total_nulls = 0;
      for (auto& cr : chunk_refs) {
        total_nulls += chunks[cr.chunk_index].null_cnt;
      }

      rmm::device_buffer null_mask;
      if (total_nulls > 0) {
        null_mask =
          cudf::create_null_mask(col_total_rows, cudf::mask_state::ALL_VALID, stream, mr_ref);

        std::vector<cuda::tae::BatchedNullMaskDesc> h_null_descs;
        std::size_t bitmask_row_offset = 0;
        for (auto& bm : block_meta) {
          auto& chunk = chunks[bm.chunk_index];
          if (chunk.null_cnt > 0) {
            auto* d_src_blk            = chunk_device_ptrs[bm.chunk_index];
            uint32_t nsp_bitmap_offset = chunks[bm.chunk_index].vector_header_size +
                                         bm.actual_data_len + 4 + bm.area_len + 4 + 24;
            h_null_descs.push_back({d_src_blk + nsp_bitmap_offset,
                                    bm.rows,
                                    static_cast<uint32_t>(bitmask_row_offset / 32)});
          }
          bitmask_row_offset += bm.rows;
        }

        if (!h_null_descs.empty()) {
          rmm::device_buffer d_null_descs(
            h_null_descs.size() * sizeof(cuda::tae::BatchedNullMaskDesc), stream, mr_ref);
          CUDF_CUDA_TRY(
            cudaMemcpyAsync(d_null_descs.data(),
                            h_null_descs.data(),
                            h_null_descs.size() * sizeof(cuda::tae::BatchedNullMaskDesc),
                            cudaMemcpyHostToDevice,
                            stream.value()));
          cuda::tae::batched_invert_null_mask(
            static_cast<cuda::tae::BatchedNullMaskDesc*>(d_null_descs.data()),
            static_cast<uint32_t>(h_null_descs.size()),
            static_cast<uint32_t*>(null_mask.data()),
            stream);
        }
      }

      auto str_col =
        cudf::make_strings_column(static_cast<cudf::size_type>(col_total_rows),
                                  std::move(offsets_col),
                                  std::move(chars_buf),
                                  total_nulls > 0 ? static_cast<cudf::size_type>(total_nulls) : 0,
                                  total_nulls > 0 ? std::move(null_mask) : rmm::device_buffer{});

      columns.push_back(std::move(str_col));
    } else {
      // Fixed-width type
      uint32_t elem_size = mo_oid_fixed_width(type_oid);
      if (elem_size == 0) elem_size = 8;  // fallback

      // Compute epoch adjustment for temporal types
      int64_t epoch_adjust = 0;
      if (is_date_type(type_oid))
        epoch_adjust = tae::MO_UNIX_EPOCH_DAYS;
      else if (is_timestamp_type(type_oid))
        epoch_adjust = tae::MO_UNIX_EPOCH_USEC;

      // Allocate output column data
      auto data_buf = rmm::device_buffer(col_total_rows * elem_size, stream, mr_ref);

      // Build batched descriptors for all blocks
      std::vector<cuda::tae::BatchedFixedDesc> h_descs;
      h_descs.reserve(chunk_refs.size());
      uint32_t max_block_rows = 0;
      std::size_t row_offset  = 0;
      for (auto& cr : chunk_refs) {
        auto& chunk = chunks[cr.chunk_index];
        h_descs.push_back({chunk_device_ptrs[cr.chunk_index] + chunk.vector_header_size,
                           chunk.row_count,
                           static_cast<uint32_t>(row_offset)});
        max_block_rows = std::max(max_block_rows, chunk.row_count);
        row_offset += chunk.row_count;
      }

      // Upload descriptors + single batched kernel launch
      rmm::device_buffer d_descs(
        h_descs.size() * sizeof(cuda::tae::BatchedFixedDesc), stream, mr_ref);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                                    h_descs.data(),
                                    h_descs.size() * sizeof(cuda::tae::BatchedFixedDesc),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      cuda::tae::batched_decode_fixed_width(
        static_cast<cuda::tae::BatchedFixedDesc*>(d_descs.data()),
        static_cast<uint32_t>(h_descs.size()),
        static_cast<uint8_t*>(data_buf.data()),
        elem_size,
        epoch_adjust,
        max_block_rows,
        stream);

      // Build null mask (batched)
      uint32_t total_nulls = 0;
      for (auto& cr : chunk_refs) {
        total_nulls += chunks[cr.chunk_index].null_cnt;
      }

      rmm::device_buffer null_mask;
      if (total_nulls > 0) {
        null_mask =
          cudf::create_null_mask(col_total_rows, cudf::mask_state::ALL_VALID, stream, mr_ref);

        std::vector<cuda::tae::BatchedNullMaskDesc> h_null_descs;
        std::size_t bitmask_row_offset = 0;
        for (auto& cr : chunk_refs) {
          auto& chunk = chunks[cr.chunk_index];
          if (chunk.null_cnt == 0) {
            bitmask_row_offset += chunk.row_count;
            continue;
          }
          auto* d_src                = chunk_device_ptrs[cr.chunk_index];
          uint32_t data_len          = chunk.row_count * elem_size;
          uint32_t nsp_bitmap_offset = chunk.vector_header_size + data_len + 4 + 0 + 4 + 24;
          h_null_descs.push_back({d_src + nsp_bitmap_offset,
                                  chunk.row_count,
                                  static_cast<uint32_t>(bitmask_row_offset / 32)});
          bitmask_row_offset += chunk.row_count;
        }

        if (!h_null_descs.empty()) {
          rmm::device_buffer d_null_descs(
            h_null_descs.size() * sizeof(cuda::tae::BatchedNullMaskDesc), stream, mr_ref);
          CUDF_CUDA_TRY(
            cudaMemcpyAsync(d_null_descs.data(),
                            h_null_descs.data(),
                            h_null_descs.size() * sizeof(cuda::tae::BatchedNullMaskDesc),
                            cudaMemcpyHostToDevice,
                            stream.value()));
          cuda::tae::batched_invert_null_mask(
            static_cast<cuda::tae::BatchedNullMaskDesc*>(d_null_descs.data()),
            static_cast<uint32_t>(h_null_descs.size()),
            static_cast<uint32_t*>(null_mask.data()),
            stream);
        }
      }

      auto col = std::make_unique<cudf::column>(
        cudf_type,
        static_cast<cudf::size_type>(col_total_rows),
        std::move(data_buf),
        total_nulls > 0 ? std::move(null_mask) : rmm::device_buffer{},
        total_nulls > 0 ? static_cast<cudf::size_type>(total_nulls) : 0);

      columns.push_back(std::move(col));
    }
  }

#ifdef SIRIUS_PROFILE
  auto const cvt_decode = std::chrono::high_resolution_clock::now();
  CUDF_CUDA_TRY(cudaEventRecord(ev_decode, stream.value()));
#endif

  // 9. Apply filter expression on GPU (if present)
  auto const& filter_expr = host_src.get_filter_expression();
#ifdef SIRIUS_PROFILE
  auto cvt_filter_compute = cvt_t0;
  auto cvt_filter_apply   = cvt_t0;
  auto cvt_proj           = cvt_t0;
#endif
  if (filter_expr && filter_expr->size() > 0) {
    // Build a temporary table to evaluate the filter against ALL columns
    auto pre_filter_table = std::make_unique<cudf::table>(std::move(columns));
    auto table_view       = pre_filter_table->view();

    SIRIUS_LOG_DEBUG("[tae_converter] applying GPU filter on {} rows, {} columns",
                     table_view.num_rows(),
                     table_view.num_columns());

    // Evaluate filter AST → boolean mask column
    auto mask = cudf::compute_column(table_view, filter_expr->back(), stream, mr_ref);
#ifdef SIRIUS_PROFILE
    cvt_filter_compute = std::chrono::high_resolution_clock::now();
    CUDF_CUDA_TRY(cudaEventRecord(ev_filter, stream.value()));
#endif

    // Apply boolean mask to keep only matching rows
    auto filtered_table = cudf::apply_boolean_mask(table_view, mask->view(), stream, mr_ref);
#ifdef SIRIUS_PROFILE
    cvt_filter_apply = std::chrono::high_resolution_clock::now();
    CUDF_CUDA_TRY(cudaEventRecord(ev_apply, stream.value()));
#endif

    SIRIUS_LOG_DEBUG("[tae_converter] filter reduced {} → {} rows",
                     table_view.num_rows(),
                     filtered_table->num_rows());

    // Release filtered columns back into the vector
    columns = filtered_table->release();
  } else {
#ifdef SIRIUS_PROFILE
    // No filter — alias events so GPU timing reports zeros for filter phases
    CUDF_CUDA_TRY(cudaEventRecord(ev_filter, stream.value()));
    CUDF_CUDA_TRY(cudaEventRecord(ev_apply, stream.value()));
#endif
  }

  // 10. Apply post-filter projection (prune filter-only columns)
  if (!post_filter_projection_ids.empty()) {
    std::vector<std::unique_ptr<cudf::column>> projected;
    projected.reserve(post_filter_projection_ids.size());
    for (auto id : post_filter_projection_ids) {
      if (id < columns.size()) { projected.push_back(std::move(columns[id])); }
    }
    columns = std::move(projected);
  }

#ifdef SIRIUS_PROFILE
  cvt_proj = std::chrono::high_resolution_clock::now();
  CUDF_CUDA_TRY(cudaEventRecord(ev_end, stream.value()));
#endif

  stream.synchronize();

#ifdef SIRIUS_PROFILE
  // Query GPU-side elapsed times (non-blocking — already synced)
  float gpu_xfer_ms = 0, gpu_lz4_ms = 0, gpu_decode_ms = 0;
  float gpu_filter_ms = 0, gpu_apply_ms = 0, gpu_tail_ms = 0;
  CUDF_CUDA_TRY(cudaEventElapsedTime(&gpu_xfer_ms, ev_pre_h2d, ev_h2d));
  CUDF_CUDA_TRY(cudaEventElapsedTime(&gpu_lz4_ms, ev_h2d, ev_lz4));
  CUDF_CUDA_TRY(cudaEventElapsedTime(&gpu_decode_ms, ev_lz4, ev_decode));
  CUDF_CUDA_TRY(cudaEventElapsedTime(&gpu_filter_ms, ev_decode, ev_filter));
  CUDF_CUDA_TRY(cudaEventElapsedTime(&gpu_apply_ms, ev_filter, ev_apply));
  CUDF_CUDA_TRY(cudaEventElapsedTime(&gpu_tail_ms, ev_apply, ev_end));
  float gpu_total_ms = 0;
  CUDF_CUDA_TRY(cudaEventElapsedTime(&gpu_total_ms, ev_pre_h2d, ev_end));

  CUDF_CUDA_TRY(cudaEventDestroy(ev_pre_h2d));
  CUDF_CUDA_TRY(cudaEventDestroy(ev_h2d));
  CUDF_CUDA_TRY(cudaEventDestroy(ev_lz4));
  CUDF_CUDA_TRY(cudaEventDestroy(ev_decode));
  CUDF_CUDA_TRY(cudaEventDestroy(ev_filter));
  CUDF_CUDA_TRY(cudaEventDestroy(ev_apply));
  CUDF_CUDA_TRY(cudaEventDestroy(ev_end));

  auto const cvt_end = std::chrono::high_resolution_clock::now();
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  SIRIUS_LOG_INFO(
    "[tae_converter] host timing: H2D={:.2f}ms LZ4={:.2f}ms decode={:.2f}ms "
    "filter_compute={:.2f}ms filter_apply={:.2f}ms "
    "proj={:.2f}ms sync={:.2f}ms total={:.2f}ms ({} bytes H2D)",
    ms(cvt_t0, cvt_h2d),
    ms(cvt_h2d, cvt_lz4),
    ms(cvt_lz4, cvt_decode),
    ms(cvt_decode, cvt_filter_compute),
    ms(cvt_filter_compute, cvt_filter_apply),
    ms(cvt_filter_apply, cvt_proj),
    ms(cvt_proj, cvt_end),
    ms(cvt_t0, cvt_end),
    linear_host.size());
  SIRIUS_LOG_INFO(
    "[tae_converter] GPU  timing: xfer={:.2f}ms lz4={:.2f}ms decode={:.2f}ms "
    "filter={:.2f}ms apply={:.2f}ms tail={:.2f}ms total={:.2f}ms",
    gpu_xfer_ms,
    gpu_lz4_ms,
    gpu_decode_ms,
    gpu_filter_ms,
    gpu_apply_ms,
    gpu_tail_ms,
    gpu_total_ms);
#endif

  auto table = std::make_unique<cudf::table>(std::move(columns));

  SIRIUS_LOG_TRACE("[tae_converter] produced GPU table: {} columns, {} rows",
                   table->num_columns(),
                   table->num_rows());

  auto result = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
  host_src.mark_h2d_complete();
  return result;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public registration
// ---------------------------------------------------------------------------
void register_tae_converters(cucascade::representation_converter_registry& registry)
{
  if (!registry.has_converter<host_tae_representation, cucascade::gpu_table_representation>()) {
    registry.register_converter<host_tae_representation, cucascade::gpu_table_representation>(
      detail::convert_host_tae_to_gpu);
  }
}

}  // namespace sirius
