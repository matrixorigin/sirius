/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/result_codec.hpp"

#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <limits>

namespace sirius::embedding {
namespace {
void cuda_check(cudaError_t status)
{
  if (status != cudaSuccess)
    throw pipeline::gpu_stream_quiescence_error(cudaGetErrorString(status));
}
void download(input_storage& destination,
              std::size_t offset,
              void const* source,
              std::size_t bytes,
              rmm::cuda_stream_view stream)
{
  destination.visit([&](std::size_t base, std::span<std::byte> block) {
    auto low = std::max(base, offset), high = std::min(base + block.size(), offset + bytes);
    if (low < high)
      cuda_check(cudaMemcpyAsync(block.data() + low - base,
                                 static_cast<std::byte const*>(source) + low - offset,
                                 high - low,
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  });
  cuda_check(cudaStreamSynchronize(stream.value()));
}
uint64_t string_offset(cudf::column_view const& col,
                       uint32_t row,
                       result_batch& scratch,
                       rmm::cuda_stream_view stream)
{
  auto offsets = cudf::strings_column_view(col).offsets();
  auto index   = static_cast<std::size_t>(col.offset()) + row;
  if (offsets.type().id() == cudf::type_id::INT64) {
    int64_t value{};
    download(*scratch.storage, 0, offsets.head<int64_t>() + index, sizeof(value), stream);
    scratch.storage->read(0, {reinterpret_cast<std::byte*>(&value), sizeof(value)});
    if (value < 0) throw failure(SIRIUS_EXECUTION_FAILED, "negative string offset");
    return value;
  }
  int32_t value{};
  download(*scratch.storage, 0, offsets.head<int32_t>() + index, sizeof(value), stream);
  scratch.storage->read(0, {reinterpret_cast<std::byte*>(&value), sizeof(value)});
  if (value < 0) throw failure(SIRIUS_EXECUTION_FAILED, "negative string offset");
  return value;
}
cudf::data_type expected(owned_column const& c)
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
std::size_t align8(std::size_t n) { return (n + 7) & ~std::size_t(7); }
sirius_input_vector column_layout(cudf::column_view const& col,
                                  owned_column const& type,
                                  uint32_t begin,
                                  uint32_t rows,
                                  std::size_t& position,
                                  result_batch& scratch,
                                  rmm::cuda_stream_view stream)
{
  sirius_input_vector vector{};
  vector.data_offset = position;
  vector.data_bytes  = static_cast<uint64_t>(rows) * input_element_size(type.oid);
  position           = align8(position + vector.data_bytes);
  vector.null_offset = position;
  vector.null_bytes  = (static_cast<uint64_t>(rows) + 63) / 64 * 8;
  position += vector.null_bytes;
  if (input_string_type(type.oid)) {
    auto low  = string_offset(col, begin, scratch, stream);
    auto high = string_offset(col, begin + rows, scratch, stream);
    if (high < low || high - low > UINT32_MAX)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native string area exceeds representation");
    vector.area_offset = position;
    vector.area_bytes  = high - low;
    position           = align8(position + vector.area_bytes);
  }
  return vector;
}
}  // namespace
result_slice_layout size_native_result(cudf::table_view table,
                                       uint32_t begin,
                                       std::span<const owned_column> schema,
                                       result_batch& scratch,
                                       rmm::cuda_stream_view stream,
                                       std::size_t target)
{
  if (table.num_columns() != static_cast<cudf::size_type>(schema.size()))
    throw failure(SIRIUS_EXECUTION_FAILED, "native result column count mismatch");
  for (std::size_t c = 0; c < schema.size(); ++c)
    if (table.column(c).type() != expected(schema[c]))
      throw failure(SIRIUS_UNSUPPORTED, "native result physical type does not match MO contract");
  auto size = [&](uint32_t rows) {
    std::size_t position{};
    for (std::size_t c = 0; c < schema.size(); ++c)
      column_layout(table.column(c), schema[c], begin, rows, position, scratch, stream);
    return position;
  };
  uint32_t available = static_cast<uint32_t>(table.num_rows()) - begin;
  if (!available) return {};
  uint32_t low = 1, high = available;
  if (size(1) > target) return {1, size(1)};
  while (low < high) {
    auto middle = low + (high - low + 1) / 2;
    if (size(middle) <= target)
      low = middle;
    else
      high = middle - 1;
  }
  return {low, size(low)};
}
void encode_native_result(cudf::table_view table,
                          uint32_t begin,
                          std::span<const owned_column> schema,
                          result_slice_layout const& layout,
                          result_batch& output,
                          result_batch& scratch,
                          rmm::cuda_stream_view stream)
{
  output.rows = layout.rows;
  std::size_t position{};
  for (std::size_t c = 0; c < schema.size(); ++c) {
    auto col     = table.column(c);
    auto& vector = output.columns[c];
    vector       = column_layout(col, schema[c], begin, layout.rows, position, scratch, stream);
    auto width   = input_element_size(schema[c].oid);
    if (!input_string_type(schema[c].oid)) {
      download(*output.storage,
               vector.data_offset,
               col.head<uint8_t>() + (static_cast<std::size_t>(col.offset()) + begin) * width,
               vector.data_bytes,
               stream);
    } else {
      auto strings = cudf::strings_column_view(col);
      auto base    = string_offset(col, begin, scratch, stream);
      if (vector.area_bytes)
        download(*output.storage,
                 vector.area_offset,
                 strings.chars_begin(stream) + base,
                 vector.area_bytes,
                 stream);
      // Bounded offset chunks; descriptors go directly to their final storage.
      auto offsets      = strings.offsets();
      auto offset_width = offsets.type().id() == cudf::type_id::INT64 ? 8u : 4u;
      for (uint32_t first = 0; first < layout.rows; first += 1024) {
        auto count = std::min<uint32_t>(1024, layout.rows - first);
        download(*scratch.storage,
                 0,
                 offsets.head<uint8_t>() +
                   (static_cast<std::size_t>(col.offset()) + begin + first) * offset_width,
                 (count + 1) * offset_width,
                 stream);
        for (uint32_t row = 0; row < count; ++row) {
          uint64_t low{}, high{};
          scratch.storage->read(row * offset_width,
                                {reinterpret_cast<std::byte*>(&low), offset_width});
          scratch.storage->read((row + 1) * offset_width,
                                {reinterpret_cast<std::byte*>(&high), offset_width});
          if (low < base || high < low || high - base > vector.area_bytes)
            throw failure(SIRIUS_EXECUTION_FAILED, "invalid native string offsets");
          std::array<std::byte, 24> value{};
          std::fill_n(value.begin(), 4, std::byte{0xff});
          uint32_t relative = low - base, length = high - low;
          std::memcpy(value.data() + 4, &relative, 4);
          std::memcpy(value.data() + 8, &length, 4);
          output.storage->write(vector.data_offset + (first + row) * 24ULL, value);
        }
      }
    }
    // Convert cuDF validity bits (1=valid) to MO words (1=null), honoring slices.
    std::array<uint32_t, 129> validity{};
    uint64_t mask_origin{};
    for (uint32_t first = 0; first < layout.rows; first += 64) {
      uint64_t nulls{};
      auto count = std::min<uint32_t>(64, layout.rows - first);
      if (col.nullable()) {
        if (first % 4096 == 0) {
          auto bit    = static_cast<uint64_t>(col.offset()) + begin + first;
          mask_origin = bit % 32;
          auto rows   = std::min<uint32_t>(4096, layout.rows - first);
          auto words  = (mask_origin + rows + 31) / 32;
          download(*scratch.storage, 0, col.null_mask() + bit / 32, words * 4, stream);
          scratch.storage->read(0, {reinterpret_cast<std::byte*>(validity.data()), words * 4});
        }
        for (uint32_t r = 0; r < count; ++r) {
          auto index = mask_origin + first % 4096 + r;
          if (!(validity[index / 32] & (1u << (index % 32)))) nulls |= uint64_t(1) << r;
        }
      }
      if (nulls && !schema[c].nullable)
        throw failure(SIRIUS_EXECUTION_FAILED, "NULL in nonnullable result column");
      output.storage->write(vector.null_offset + first / 8,
                            {reinterpret_cast<std::byte const*>(&nulls), sizeof(nulls)});
      if (nulls) {
        std::array<std::byte, 256> zeros{};
        for (uint32_t r = 0; r < count; ++r) {
          if (!(nulls & (uint64_t(1) << r))) continue;
          auto at = vector.data_offset + static_cast<uint64_t>(first + r) * width;
          if (input_string_type(schema[c].oid)) {
            uint32_t offset{}, length{};
            output.storage->read(at + 4, {reinterpret_cast<std::byte*>(&offset), 4});
            output.storage->read(at + 8, {reinterpret_cast<std::byte*>(&length), 4});
            for (std::size_t done = 0; done < length; done += zeros.size())
              output.storage->write(
                vector.area_offset + offset + done,
                std::span(zeros).first(std::min<std::size_t>(zeros.size(), length - done)));
          }
          output.storage->write(at, std::span(zeros).first(width));
        }
      }
      if (schema[c].oid == 50 || schema[c].oid == 52) {
        for (uint32_t r = 0; r < count; ++r) {
          auto at = vector.data_offset + static_cast<uint64_t>(first + r) * width;
          int64_t value{};
          if (!(nulls & (uint64_t(1) << r))) {
            output.storage->read(at, {reinterpret_cast<std::byte*>(&value), width});
            if (width == 4) value = static_cast<int32_t>(value);
            auto epoch = width == 4 ? int64_t(719162) : int64_t(62135596800000000ULL);
            if (value > std::numeric_limits<int64_t>::max() - epoch)
              throw failure(SIRIUS_EXECUTION_FAILED, "native temporal result overflow");
            value += epoch;
            if (width == 4 && (value < INT32_MIN || value > INT32_MAX))
              throw failure(SIRIUS_EXECUTION_FAILED, "native date result overflow");
          }
          output.storage->write(at, {reinterpret_cast<std::byte const*>(&value), width});
        }
      }
    }
  }
}
}  // namespace sirius::embedding
