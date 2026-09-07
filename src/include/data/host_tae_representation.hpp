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

#pragma once

// sirius
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <tae/tae_format.hpp>

// cucascade
#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// CUDA
#include <cuda_runtime.h>

// standard library
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius {

// Optional one-pass ownership carried only by streamed MO input. The
// representation marks it after a successful synchronized H2D conversion; its
// destructor then releases the source generation and retained host reservation.
class host_tae_input_lease {
 public:
  virtual ~host_tae_input_lease()           = default;
  virtual void mark_h2d_complete() noexcept = 0;
};

/**
 * @brief RAII wrapper for host memory that uses CUDA pinned (page-locked)
 * allocation above a size threshold, falling back to regular heap allocation
 * for small buffers where cudaMallocHost overhead would dominate.
 *
 * Pinned memory enables truly asynchronous cudaMemcpyAsync transfers,
 * avoiding the internal staging copy that pageable memory requires.
 * Contiguous allocations expose data()/size(); pooled allocations use bounded
 * copy_from/copy_to_device operations across their constituent blocks.
 */
struct pinned_host_buffer {
  static constexpr std::size_t PINNED_THRESHOLD = std::size_t(64) << 20;  // 64 MB

  pinned_host_buffer() = default;

  explicit pinned_host_buffer(std::size_t n)
    : _size(n), _capacity(n), _pinned(n >= PINNED_THRESHOLD)
  {
    if (n > 0) {
      if (_pinned) {
        auto err = cudaMallocHost(&_ptr, n);
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                                   cudaGetErrorString(err));
        }
      } else {
        _ptr = new uint8_t[n];
      }
    }
  }

  // Borrow bounded blocks from the same pool used by the other scan sources.
  // The allocation must be destroyed before its reservation and memory space.
  pinned_host_buffer(std::size_t n,
                     cucascade::memory::fixed_size_host_memory_resource& pool,
                     std::unique_ptr<cucascade::memory::reservation> reservation)
    : _size(n), _capacity(n), _pinned(true), _reservation(std::move(reservation))
  {
    if (!_reservation || _reservation->size() < n) {
      throw std::invalid_argument("pooled host buffer requires its complete reservation");
    }
    _blocks = pool.allocate_multiple_blocks(n, _reservation.get());
  }

  ~pinned_host_buffer()
  {
    if (_ptr) {
      if (_pinned)
        cudaFreeHost(_ptr);
      else
        delete[] static_cast<uint8_t*>(_ptr);
    }
  }

  pinned_host_buffer(pinned_host_buffer&& o) noexcept
    : _ptr(o._ptr),
      _size(o._size),
      _capacity(o._capacity),
      _pinned(o._pinned),
      _reservation(std::move(o._reservation)),
      _blocks(std::move(o._blocks))
  {
    o._ptr      = nullptr;
    o._size     = 0;
    o._capacity = 0;
  }

  pinned_host_buffer& operator=(pinned_host_buffer&& o) noexcept
  {
    if (this != &o) {
      _blocks.reset();
      _reservation.reset();
      if (_ptr) {
        if (_pinned)
          cudaFreeHost(_ptr);
        else
          delete[] static_cast<uint8_t*>(_ptr);
      }
      _ptr         = o._ptr;
      _size        = o._size;
      _capacity    = o._capacity;
      _pinned      = o._pinned;
      _reservation = std::move(o._reservation);
      _blocks      = std::move(o._blocks);
      o._ptr       = nullptr;
      o._size      = 0;
      o._capacity  = 0;
    }
    return *this;
  }

  pinned_host_buffer(pinned_host_buffer const&)            = delete;
  pinned_host_buffer& operator=(pinned_host_buffer const&) = delete;

  [[nodiscard]] uint8_t* data()
  {
    if (_blocks) { throw std::logic_error("pooled host buffer is not contiguous"); }
    return static_cast<uint8_t*>(_ptr);
  }
  [[nodiscard]] uint8_t const* data() const
  {
    if (_blocks) { throw std::logic_error("pooled host buffer is not contiguous"); }
    return static_cast<uint8_t const*>(_ptr);
  }
  [[nodiscard]] std::size_t size() const { return _size; }
  [[nodiscard]] std::size_t capacity() const { return _capacity; }
  [[nodiscard]] bool is_pinned() const { return _pinned; }

  void copy_from(std::size_t offset, void const* source, std::size_t bytes)
  {
    if (offset > _capacity || bytes > _capacity - offset) {
      throw std::out_of_range("host buffer write exceeds capacity");
    }
    if (bytes == 0) { return; }
    if (!_blocks) {
      std::memcpy(static_cast<uint8_t*>(_ptr) + offset, source, bytes);
      return;
    }
    auto* input           = static_cast<uint8_t const*>(source);
    const auto block_size = _blocks->block_size();
    while (bytes != 0) {
      const auto in_block = offset % block_size;
      const auto count    = std::min(bytes, block_size - in_block);
      std::memcpy(_blocks->at(offset / block_size).data() + in_block, input, count);
      input += count;
      offset += count;
      bytes -= count;
    }
  }

  cudaError_t copy_to_device(void* destination, rmm::cuda_stream_view stream) const
  {
    if (_size == 0) { return cudaSuccess; }
    if (!_blocks) {
      return cudaMemcpyAsync(destination, _ptr, _size, cudaMemcpyHostToDevice, stream.value());
    }
    auto* output          = static_cast<uint8_t*>(destination);
    std::size_t remaining = _size;
    for (std::size_t i = 0; remaining != 0; ++i) {
      const auto block = _blocks->at(i);
      const auto count = std::min(remaining, block.size());
      auto status =
        cudaMemcpyAsync(output, block.data(), count, cudaMemcpyHostToDevice, stream.value());
      if (status != cudaSuccess) { return status; }
      output += count;
      remaining -= count;
    }
    return cudaSuccess;
  }

  void set_logical_size(std::size_t size)
  {
    if (size > _capacity) { throw std::out_of_range("pinned host buffer size exceeds capacity"); }
    _size = size;
  }

 private:
  void* _ptr            = nullptr;
  std::size_t _size     = 0;
  std::size_t _capacity = 0;
  bool _pinned          = false;
  // Blocks refer to the reservation arena, so return them before releasing it.
  std::unique_ptr<cucascade::memory::reservation> _reservation;
  std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation>
    _blocks;
};

/**
 * @brief Host representation of compressed TAE column data.
 *
 * Each instance holds one TAE object's worth of projected column data in
 * LZ4-compressed form (CRC already stripped). The converter pipeline transfers
 * this data to GPU, decompresses with nvCOMP batch LZ4, then decodes MO types
 * to cuDF columns via custom CUDA kernels.
 *
 * Layout per column chunk in the buffer:
 *   [IOEntryHeader 4B] [compressed payload ...]
 * The Extent in column_meta tells offset+length for the compressed data,
 * and origin_size for the decompressed size.
 */
class host_tae_representation : public cucascade::idata_representation {
  using translated_expression = gpu_expression_translator::translated_expression;

 public:
  /**
   * @brief Metadata for a single column chunk within this representation.
   */
  struct column_chunk_info {
    uint16_t column_idx;                    ///< Column ordinal in the TAE object
    tae::MOTypeOid type_oid;                ///< MO type for decode
    int32_t width;                          ///< DECIMAL precision
    int32_t scale;                          ///< DECIMAL scale
    tae::Extent extent;                     ///< Compressed data location
    uint32_t null_cnt;                      ///< Number of nulls in this chunk
    uint32_t row_count;                     ///< Number of rows in the block
    std::size_t pinned_offset;              ///< Byte offset into host buffer
    std::size_t pinned_length;              ///< Byte count in host buffer (compressed)
    std::uint32_t vector_header_size = 29;  ///< Offset from decoded chunk to vector data
  };

  /**
   * @brief Constructs a host_tae_representation.
   *
   * @param memory_space  The memory space for the target GPU.
   * @param host_data     Pinned host buffer holding compressed column data.
   * @param chunks        Per-column metadata for the converter to decode.
   * @param total_rows    Total row count across all blocks in this object.
   * @param compressed_bytes  Total compressed bytes in host_data.
   * @param uncompressed_bytes  Sum of all origin_size values.
   * @param filter_expression  Optional pushdown filter.
   * @param post_filter_projection_ids  Column indices surviving filter.
   */
  host_tae_representation(cucascade::memory::memory_space* memory_space,
                          pinned_host_buffer host_data,
                          std::vector<column_chunk_info> chunks,
                          std::size_t total_rows,
                          std::size_t compressed_bytes,
                          std::size_t uncompressed_bytes,
                          std::shared_ptr<translated_expression> filter_expression = nullptr,
                          std::vector<std::size_t> post_filter_projection_ids      = {},
                          std::unique_ptr<host_tae_input_lease> input_lease        = nullptr);

  // idata_representation interface
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override;
  [[nodiscard]] std::size_t get_size_in_bytes() const override;
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override;

  // Accessors
  [[nodiscard]] auto const& get_host_data() const { return _host_data; }
  [[nodiscard]] auto const& get_column_chunks() const { return _chunks; }
  [[nodiscard]] std::size_t get_total_rows() const { return _total_rows; }
  void mark_h2d_complete() noexcept
  {
    if (_input_lease) { _input_lease->mark_h2d_complete(); }
  }

  [[nodiscard]] std::shared_ptr<translated_expression> const& get_filter_expression() const
  {
    return _filter_expression;
  }

  [[nodiscard]] std::vector<std::size_t> const& get_post_filter_projection_ids() const
  {
    return _post_filter_projection_ids;
  }

 private:
  host_tae_representation(const host_tae_representation& other);

  // Declared before the raw representation fields so it is destroyed last.
  // This guarantees the old host bytes are gone before the source generation
  // is released and the prefetched frame can be pulled.
  std::unique_ptr<host_tae_input_lease> _input_lease;
  std::shared_ptr<pinned_host_buffer> _host_data;

  std::vector<column_chunk_info> _chunks;
  std::size_t _total_rows;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::shared_ptr<translated_expression> _filter_expression;
  std::vector<std::size_t> _post_filter_projection_ids;
};

}  // namespace sirius
