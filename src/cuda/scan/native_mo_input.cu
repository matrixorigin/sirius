/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/native_gpu.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/scan.h>

namespace sirius::embedding {
namespace {
struct device_slice {
  const uint8_t* base;
  sirius_input_vector column;
  uint32_t begin, rows, output;
};
__device__ uint32_t u32(const uint8_t* p)
{
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
__device__ bool is_null(device_slice const& d, uint32_t row)
{
  if (d.column.vector_class == SIRIUS_VECTOR_NULL) return true;
  if (d.column.vector_class == SIRIUS_VECTOR_CONSTANT) row = 0;
  return row / 8 < d.column.null_bytes &&
         (d.base[d.column.null_offset + row / 8] & (1u << (row % 8)));
}
__device__ const uint8_t* string_data(device_slice const& d, uint32_t row, uint32_t& length)
{
  if (is_null(d, row)) {
    length = 0;
    return nullptr;
  }
  if (d.column.vector_class == SIRIUS_VECTOR_CONSTANT) row = 0;
  auto p = d.base + d.column.data_offset + uint64_t(row) * 24;
  if (p[0] <= 23) {
    length = p[0];
    return p + 1;
  }
  length = u32(p + 8);
  return d.base + d.column.area_offset + u32(p + 4);
}
__global__ void decode(device_slice const* slices,
                       uint8_t* values,
                       uint32_t* validity,
                       uint32_t width,
                       uint32_t oid,
                       int32_t* lengths)
{
  auto const& d = slices[blockIdx.y];
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < d.rows;
       r += blockDim.x * gridDim.x) {
    auto row = d.begin + r, out = d.output + r;
    bool null = is_null(d, row);
    if (null) atomicAnd(validity + out / 32, ~(1u << (out % 32)));
    if (lengths) {
      uint32_t length;
      string_data(d, row, length);
      lengths[out] = length;
    } else {
      auto srcrow = d.column.vector_class == SIRIUS_VECTOR_CONSTANT ? 0 : row;
      auto src    = d.base + d.column.data_offset + uint64_t(srcrow) * width;
      auto dst    = values + uint64_t(out) * width;
      for (uint32_t b = 0; b < width; ++b)
        dst[b] = null ? 0 : src[b];
      if (!null && oid == 50) *reinterpret_cast<uint32_t*>(dst) -= 719162u;
      if (!null && oid == 52) *reinterpret_cast<uint64_t*>(dst) -= 62135596800000000ULL;
    }
  }
}
__global__ void scatter(device_slice const* slices, int32_t const* offsets, uint8_t* chars)
{
  auto const& d = slices[blockIdx.y];
  for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < d.rows;
       r += blockDim.x * gridDim.x) {
    uint32_t length;
    auto src = string_data(d, d.begin + r, length);
    auto dst = chars + offsets[d.output + r];
    for (uint32_t b = 0; b < length; ++b)
      dst[b] = src[b];
  }
}
cudf::data_type type(sirius_input_column const& c)
{
  using id = cudf::type_id;
  switch (c.oid) {
    case 10: return cudf::data_type{id::BOOL8};
    case 20: return cudf::data_type{id::INT8};
    case 21: return cudf::data_type{id::INT16};
    case 22: return cudf::data_type{id::INT32};
    case 23: return cudf::data_type{id::INT64};
    case 25: return cudf::data_type{id::UINT8};
    case 26: return cudf::data_type{id::UINT16};
    case 27: return cudf::data_type{id::UINT32};
    case 28: return cudf::data_type{id::UINT64};
    case 30: return cudf::data_type{id::FLOAT32};
    case 31: return cudf::data_type{id::FLOAT64};
    case 32: return cudf::data_type{id::DECIMAL64, -c.scale};
    case 33: return cudf::data_type{id::DECIMAL128, -c.scale};
    case 50: return cudf::data_type{id::TIMESTAMP_DAYS};
    case 52: return cudf::data_type{id::TIMESTAMP_MICROSECONDS};
    default: return cudf::data_type{id::STRING};
  }
}
struct conversion_owner {
  std::vector<rmm::device_buffer> mirrors, descriptors;
  std::vector<std::vector<device_slice>> host_descriptors;
  std::vector<std::unique_ptr<cudf::column>> columns, offsets;
};
}  // namespace
std::unique_ptr<cudf::table> convert_native_input(input_unit const& unit,
                                                  std::span<const sirius_input_column> schema,
                                                  cucascade::memory::memory_space const& gpu,
                                                  rmm::cuda_stream_view stream)
{
  auto owner = std::make_unique<conversion_owner>();
  auto mr    = gpu.get_default_allocator();
  try {
    if (!unit.rows) {
      if (schema.empty())
        owner->columns.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT8}));
      for (auto const& c : schema)
        owner->columns.push_back(cudf::make_empty_column(type(c)));
      return std::make_unique<cudf::table>(std::move(owner->columns));
    }
    for (auto const& slice : unit.slices) {
      auto& mirror = owner->mirrors.emplace_back(slice.batch->payload_bytes, stream, mr);
      slice.batch->storage->visit([&](std::size_t offset, std::span<std::byte> block) {
        if (offset >= mirror.size()) return;
        block = block.first(std::min(block.size(), mirror.size() - offset));
        CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<std::byte*>(mirror.data()) + offset,
                                      block.data(),
                                      block.size(),
                                      cudaMemcpyHostToDevice,
                                      stream.value()));
      });
    }
    if (schema.empty()) {
      auto carrier = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT8}, unit.rows, cudf::mask_state::UNALLOCATED, stream, mr);
      CUDF_CUDA_TRY(
        cudaMemsetAsync(carrier->mutable_view().data<uint8_t>(), 0, unit.rows, stream.value()));
      owner->columns.push_back(std::move(carrier));
    }
    owner->host_descriptors.resize(schema.size());
    for (std::size_t c = 0; c < schema.size(); ++c) {
      auto& host      = owner->host_descriptors[c];
      uint32_t output = 0;
      for (std::size_t s = 0; s < unit.slices.size(); ++s) {
        auto const& slice = unit.slices[s];
        host.push_back({static_cast<const uint8_t*>(owner->mirrors[s].data()),
                        slice.batch->columns[c],
                        slice.begin,
                        slice.rows,
                        output});
        output += slice.rows;
      }
      auto& desc = owner->descriptors.emplace_back(host.size() * sizeof(device_slice), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        desc.data(), host.data(), desc.size(), cudaMemcpyHostToDevice, stream.value()));
      auto d = static_cast<const device_slice*>(desc.data());
      dim3 grid(128, host.size());
      if (!input_string_type(schema[c].oid)) {
        auto col = cudf::make_fixed_width_column(
          type(schema[c]), unit.rows, cudf::mask_state::ALL_VALID, stream, mr);
        owner->columns.push_back(std::move(col));
        auto view = owner->columns.back()->mutable_view();
        decode<<<grid, 256, 0, stream.value()>>>(d,
                                                 view.data<uint8_t>(),
                                                 view.null_mask(),
                                                 input_element_size(schema[c].oid),
                                                 schema[c].oid,
                                                 nullptr);
        CUDF_CUDA_TRY(cudaGetLastError());
        owner->columns.back()->set_null_count(unit.nulls[c]);
      } else {
        auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                 unit.rows + 1,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
        auto ptr     = offsets->mutable_view().data<int32_t>();
        owner->offsets.push_back(std::move(offsets));
        auto mask = cudf::create_null_mask(unit.rows, cudf::mask_state::ALL_VALID, stream, mr);
        rmm::device_buffer chars(unit.chars[c], stream, mr);
        // Install owners before launching kernels so an exception cannot free
        // mask/chars before the cleanup synchronization below.
        auto col = cudf::make_strings_column(unit.rows,
                                             std::move(owner->offsets.back()),
                                             std::move(chars),
                                             unit.nulls[c],
                                             std::move(mask));
        owner->columns.push_back(std::move(col));
        auto view = owner->columns.back()->mutable_view();
        CUDF_CUDA_TRY(cudaMemsetAsync(ptr + unit.rows, 0, sizeof(int32_t), stream.value()));
        decode<<<grid, 256, 0, stream.value()>>>(d, nullptr, view.null_mask(), 0, 0, ptr);
        CUDF_CUDA_TRY(cudaGetLastError());
        thrust::exclusive_scan(rmm::exec_policy(stream, mr), ptr, ptr + unit.rows + 1, ptr);
        scatter<<<grid, 256, 0, stream.value()>>>(d, ptr, view.data<uint8_t>());
        CUDF_CUDA_TRY(cudaGetLastError());
      }
    }
    stream.synchronize();
    return std::make_unique<cudf::table>(std::move(owner->columns));
  } catch (...) {
    auto original = std::current_exception();
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error(
        "native MO decode could not prove stream quiescence");
    }
    std::rethrow_exception(original);
  }
}
}  // namespace sirius::embedding
