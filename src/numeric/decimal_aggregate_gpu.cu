/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_aggregate_gpu.hpp"
#include "numeric/decimal_error.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <optional>

namespace sirius::mo_decimal {
namespace {
struct operand {
  uint8_t const* data{};
  uint64_t const* limbs[4]{};
  cudf::bitmask_type const* mask{};
  cudf::size_type offset{};
  uint32_t bytes{};
  __device__ bool valid(cudf::size_type row) const
  {
    auto bit = row + offset;
    return !mask || ((mask[bit / 32] >> (bit % 32)) & 1);
  }
  __device__ coefficient value(cudf::size_type row) const
  {
    if (bytes != 32) return load_coefficient(data + uint64_t(row + offset) * bytes, bytes);
    coefficient value;
    for (int i = 0; i < 4; ++i) {
      auto word              = limbs[3 - i][row + offset];
      value.words[2 * i]     = word;
      value.words[2 * i + 1] = word >> 32;
    }
    return value;
  }
};
operand view(cudf::column_view column, decimal_type type)
{
  operand result;
  result.bytes  = type.bytes();
  result.offset = column.offset();
  result.mask   = column.null_mask();
  if (type.bits != 256)
    result.data = column.head<uint8_t>();
  else
    for (int i = 0; i < 4; ++i)
      result.limbs[i] = column.child(i).head<uint64_t>() + column.child(i).offset();
  return result;
}
struct destination {
  uint8_t* data{};
  uint64_t* limbs[4]{};
  uint32_t bytes{};
  __device__ void store(coefficient const& value, cudf::size_type row) const
  {
    if (bytes != 32)
      store_coefficient(value, data + uint64_t(row) * bytes, bytes);
    else
      for (int i = 0; i < 4; ++i)
        limbs[3 - i][row] = uint64_t(value.words[2 * i]) | (uint64_t(value.words[2 * i + 1]) << 32);
  }
};
destination output_view(cudf::mutable_column_view column, uint32_t bytes)
{
  destination result;
  result.bytes = bytes;
  if (bytes != 32)
    result.data = column.data<uint8_t>();
  else
    for (int i = 0; i < 4; ++i)
      result.limbs[i] = column.child(i).data<uint64_t>();
  return result;
}
constexpr int block_size = 256, tile_rows = 4096;
__global__ void tile_prefix(cudf::size_type const* offsets,
                            cudf::size_type groups,
                            int64_t* tile_offsets)
{
  for (int64_t group = blockIdx.x * blockDim.x + threadIdx.x; group <= groups;
       group += blockDim.x * gridDim.x)
    // For every segment, floor(end/T)-floor(start/T)+1 >= ceil(length/T).
    // Extra tiles are identities. This bounded prefix needs neither a device
    // scan nor a host rendezvous; subtract the origin to support subranges.
    tile_offsets[group] = (int64_t(offsets[group]) - offsets[0]) / tile_rows + group;
}
__global__ void reduce_tiles(aggregate_op op,
                             operand values,
                             uint16_t bits,
                             int64_t const* counts,
                             cudf::size_type const* order,
                             cudf::size_type const* offsets,
                             int64_t const* tile_offsets,
                             cudf::size_type groups,
                             int64_t tiles,
                             aggregate_state* partials)
{
  __shared__ aggregate_state states[block_size];
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    cudf::size_type lo = 0, hi = groups;
    while (lo + 1 < hi) {
      auto mid = lo + (hi - lo) / 2;
      if (tile_offsets[mid] <= tile)
        lo = mid;
      else
        hi = mid;
    }
    auto start = int64_t(offsets[lo]) + int64_t(tile - tile_offsets[lo]) * tile_rows;
    auto end   = start + tile_rows;
    if (end > offsets[lo + 1]) end = offsets[lo + 1];
    aggregate_state state;
    for (auto position = start + threadIdx.x; position < end; position += block_size) {
      auto row = order ? order[position] : position;
      if (values.valid(row))
        state = merge_aggregate(
          op, state, {values.value(row), counts ? counts[row] : 1, decimal_error::none}, bits);
    }
    states[threadIdx.x] = state;
    __syncthreads();
    for (int stride = block_size / 2; stride; stride /= 2) {
      if (threadIdx.x < stride)
        states[threadIdx.x] =
          merge_aggregate(op, states[threadIdx.x], states[threadIdx.x + stride], bits);
      __syncthreads();
    }
    if (!threadIdx.x) partials[tile] = states[0];
    __syncthreads();
  }
}
__global__ void reduce_groups(aggregate_op op,
                              uint16_t bits,
                              aggregate_state const* partials,
                              int64_t const* offsets,
                              cudf::size_type groups,
                              destination output,
                              int64_t* counts,
                              uint8_t* errors)
{
  __shared__ aggregate_state states[block_size];
  for (int64_t group = blockIdx.x; group < groups; group += gridDim.x) {
    aggregate_state state;
    for (int64_t tile = int64_t(offsets[group]) + threadIdx.x; tile < offsets[group + 1];
         tile += block_size)
      state = merge_aggregate(op, state, partials[tile], bits);
    states[threadIdx.x] = state;
    __syncthreads();
    for (int stride = block_size / 2; stride; stride /= 2) {
      if (threadIdx.x < stride)
        states[threadIdx.x] =
          merge_aggregate(op, states[threadIdx.x], states[threadIdx.x + stride], bits);
      __syncthreads();
    }
    if (!threadIdx.x) {
      output.store(states[0].value, group);
      counts[group] = states[0].count;
      if (errors) errors[group] = static_cast<uint8_t>(states[0].error);
    }
    __syncthreads();
  }
}
__global__ void finalize_rows(aggregate_op op,
                              operand values,
                              int64_t const* counts,
                              decimal_type input,
                              decimal_type output,
                              cudf::size_type rows,
                              destination destination,
                              cudf::bitmask_type* mask,
                              uint32_t* valid,
                              uint8_t* errors)
{
  for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    auto result =
      finalize_aggregate(op, {values.value(row), counts[row], decimal_error::none}, input, output);
    destination.store(result.value, row);
    errors[row] = static_cast<uint8_t>(result.error);
    if (result.valid) {
      atomicOr(mask + row / 32, uint32_t(1) << (row % 32));
      atomicAdd(valid, 1u);
    }
  }
}
__global__ void key_rows(
  operand values, uint8_t scale, cudf::size_type rows, destination output, uint8_t* scales)
{
  for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    auto key =
      values.valid(row) ? normalize_key(values.value(row), scale) : normalized_decimal_key{};
    output.store(key.value, row);
    scales[row] = key.scale;
  }
}
struct child_validity {
  cudf::bitmask_type const* masks[4]{};
  cudf::size_type offsets[4]{};
};
__global__ void check_children(operand values,
                               child_validity children,
                               cudf::size_type rows,
                               uint32_t* bad)
{
  for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x)
    if (values.valid(row)) {
      for (int i = 0; i < 4; ++i) {
        auto bit  = row + values.offset + children.offsets[i];
        auto mask = children.masks[i];
        if (mask && !(mask[bit / 32] & (uint32_t(1) << (bit % 32)))) atomicExch(bad, 1u);
      }
    }
}
__global__ void group_offsets(cudf::size_type const* labels,
                              cudf::size_type const* order,
                              cudf::size_type rows,
                              cudf::size_type* offsets,
                              cudf::size_type groups)
{
  for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    auto label = labels[order[row]];
    if (!row || labels[order[row - 1]] != label) offsets[label] = row;
  }
  if (!blockIdx.x && !threadIdx.x) offsets[groups] = rows;
}
template <class Owner>
[[noreturn]] void rethrow_quiescent(std::unique_ptr<Owner>& owner, rmm::cuda_stream_view stream)
{
  auto failure = std::current_exception();
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
    (void)owner.release();
    throw pipeline::gpu_stream_quiescence_error(
      "MO aggregate/key buffers could not prove quiescence");
  }
  std::rethrow_exception(failure);
}
}  // namespace
aggregate_columns reduce_aggregate_columns(aggregate_op op,
                                           cudf::column_view values,
                                           decimal_type input,
                                           uint16_t state_bits,
                                           cudf::column_view counts,
                                           cudf::column_view order,
                                           cudf::size_type const* offsets,
                                           cudf::size_type groups,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  values = canonical_decimal_view(values, input, stream, mr);
  if (!decimal_column_matches(values, input) || (state_bits != 128 && state_bits != 256) ||
      state_bits < input.bits || op > aggregate_op::max || groups < 0 || !offsets ||
      (counts.type().id() != cudf::type_id::EMPTY &&
       (counts.type().id() != cudf::type_id::INT64 || counts.size() != values.size() ||
        counts.has_nulls())) ||
      (order.type().id() != cudf::type_id::EMPTY &&
       (order.type().id() != cudf::type_id::INT32 || order.size() != values.size() ||
        order.has_nulls())))
    throw std::invalid_argument("invalid MO aggregate reduction inputs");
  struct owners {
    aggregate_columns result;
    std::unique_ptr<rmm::device_uvector<int64_t>> tile_offsets;
    std::unique_ptr<rmm::device_uvector<aggregate_state>> partials;
    std::optional<decimal_column_result> errors;
    int64_t single_offsets[2]{};
  };
  auto owner = std::make_unique<owners>();
  try {
    // At most INT32_MAX initial rows: an accumulator at least 32 bits
    // wider than the signed physical input contains every subset. MIN/MAX
    // are also infallible. This skips error polling only, never publication
    // precision validation, and depends on domains rather than sampled data.
    bool proven =
      counts.type().id() == cudf::type_id::EMPTY &&
      (state_bits >= input.bits + 32 || op == aggregate_op::min || op == aggregate_op::max);
    decimal_type state_type{
      state_bits, static_cast<uint8_t>(state_bits == 128 ? 38 : 76), input.scale};
    owner->result.values =
      make_decimal_column(state_type, groups, cudf::mask_state::UNALLOCATED, stream, mr);
    owner->result.counts = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, groups, cudf::mask_state::UNALLOCATED, stream, mr);
    owner->errors.emplace(decimal_column_result{
      nullptr, rmm::device_uvector<uint8_t>(proven ? 0 : groups, stream, mr)});
    if (groups) {
      owner->tile_offsets =
        std::make_unique<rmm::device_uvector<int64_t>>(int64_t(groups) + 1, stream, mr);
      int64_t tiles{};
      if (groups == 1) {
        // Empty extra tiles are identities even for a subrange. No device
        // prefix scan or host tile-count rendezvous is needed for one group.
        tiles                    = (int64_t(values.size()) + tile_rows - 1) / tile_rows;
        owner->single_offsets[1] = tiles;
        CUDF_CUDA_TRY(cudaMemcpyAsync(owner->tile_offsets->data(),
                                      owner->single_offsets,
                                      sizeof(owner->single_offsets),
                                      cudaMemcpyHostToDevice,
                                      stream.value()));
      } else {
        tiles = int64_t(values.size()) / tile_rows + groups;
        tile_prefix<<<128, block_size, 0, stream.value()>>>(
          offsets, groups, owner->tile_offsets->data());
        CUDF_CUDA_TRY(cudaGetLastError());
      }
      owner->partials = std::make_unique<rmm::device_uvector<aggregate_state>>(tiles, stream, mr);
      if (tiles)
        reduce_tiles<<<std::min<int64_t>(tiles, 4096), block_size, 0, stream.value()>>>(
          op,
          view(values, input),
          state_bits,
          counts.type().id() == cudf::type_id::EMPTY ? nullptr : counts.data<int64_t>(),
          order.type().id() == cudf::type_id::EMPTY ? nullptr : order.data<cudf::size_type>(),
          offsets,
          owner->tile_offsets->data(),
          groups,
          tiles,
          owner->partials->data());
      reduce_groups<<<std::min<cudf::size_type>(groups, 4096), block_size, 0, stream.value()>>>(
        op,
        state_bits,
        owner->partials->data(),
        owner->tile_offsets->data(),
        groups,
        output_view(owner->result.values->mutable_view(), state_type.bytes()),
        owner->result.counts->mutable_view().data<int64_t>(),
        proven ? nullptr : owner->errors->errors.data());
      CUDF_CUDA_TRY(cudaGetLastError());
      if (!proven) {
        auto error = column_error(*owner->errors, stream, mr);
        if (error != decimal_error::none) throw numeric_error(error);
      }
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    return std::move(owner->result);
  } catch (...) {
    rethrow_quiescent(owner, stream);
  }
}
std::unique_ptr<cudf::column> finalize_aggregate_columns(aggregate_op op,
                                                         cudf::column_view values,
                                                         cudf::column_view counts,
                                                         decimal_type input,
                                                         decimal_type output,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (values.size() != counts.size() || counts.type().id() != cudf::type_id::INT64 ||
      counts.has_nulls() || !input.valid() || !output.valid())
    throw std::invalid_argument("invalid MO aggregate finalization inputs");
  struct owners {
    std::optional<decimal_column_result> result;
    std::unique_ptr<rmm::device_uvector<uint32_t>> valid;
    uint32_t host_valid{};
  };
  auto owner = std::make_unique<owners>();
  try {
    auto rows = values.size();
    owner->result.emplace(
      decimal_column_result{nullptr, rmm::device_uvector<uint8_t>(0, stream, mr)});
    owner->result->values =
      make_decimal_column(output, rows, cudf::mask_state::ALL_NULL, stream, mr);
    owner->result->errors = rmm::device_uvector<uint8_t>(rows, stream, mr);
    owner->valid          = std::make_unique<rmm::device_uvector<uint32_t>>(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(owner->valid->data(), 0, sizeof(uint32_t), stream.value()));
    auto bits = values.type().id() == cudf::type_id::STRUCT ? 256 : 128;
    decimal_type state_type{
      static_cast<uint16_t>(bits), static_cast<uint8_t>(bits == 256 ? 76 : 38), input.scale};
    if (!decimal_column_matches(values, state_type))
      throw std::invalid_argument("invalid MO aggregate state carrier");
    if (rows)
      finalize_rows<<<128, block_size, 0, stream.value()>>>(
        op,
        view(values, state_type),
        counts.data<int64_t>(),
        input,
        output,
        rows,
        output_view(owner->result->values->mutable_view(), output.bytes()),
        owner->result->values->mutable_view().null_mask(),
        owner->valid->data(),
        owner->result->errors.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    auto error = column_error(*owner->result, stream, mr);
    if (error != decimal_error::none) throw numeric_error(error);
    CUDF_CUDA_TRY(cudaMemcpyAsync(&owner->host_valid,
                                  owner->valid->data(),
                                  sizeof(owner->host_valid),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    owner->result->values->set_null_count(rows - owner->host_valid);
    return std::move(owner->result->values);
  } catch (...) {
    rethrow_quiescent(owner, stream);
  }
}
std::unique_ptr<cudf::column> make_equality_key(cudf::column_view values,
                                                decimal_type type,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  values = canonical_decimal_view(values, type, stream, mr);
  if (!decimal_column_matches(values, type))
    throw std::invalid_argument("invalid MO decimal equality key");
  struct owners {
    std::unique_ptr<cudf::column> limbs, scales, result;
  };
  auto owner = std::make_unique<owners>();
  try {
    owner->limbs =
      make_decimal_column({256, 76, 0}, values.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    owner->scales = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8},
                                              values.size(),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
    if (values.size())
      key_rows<<<128, block_size, 0, stream.value()>>>(
        view(values, type),
        type.scale,
        values.size(),
        output_view(owner->limbs->mutable_view(), 32),
        owner->scales->mutable_view().data<uint8_t>());
    CUDF_CUDA_TRY(cudaGetLastError());
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    auto contents = owner->limbs->release();
    contents.children.push_back(std::move(owner->scales));
    owner->result = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                                   values.size(),
                                                   rmm::device_buffer{},
                                                   cudf::copy_bitmask(values, stream, mr),
                                                   values.null_count(),
                                                   std::move(contents.children));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    return std::move(owner->result);
  } catch (...) {
    rethrow_quiescent(owner, stream);
  }
}
std::unique_ptr<cudf::column> restore_decimal_validity(std::unique_ptr<cudf::column> column,
                                                       decimal_type type,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  if (type.bits != 256) {
    if (!decimal_column_matches(column->view(), type))
      throw std::invalid_argument("MO decimal carrier mismatch");
    return column;
  }
  struct owners {
    std::unique_ptr<cudf::column> column;
    std::unique_ptr<rmm::device_uvector<uint32_t>> bad;
    uint32_t host_bad{};
  };
  auto owner    = std::make_unique<owners>();
  owner->column = std::move(column);
  try {
    auto col = owner->column->view();
    if (col.type().id() != cudf::type_id::STRUCT || col.num_children() != 4 || col.offset())
      throw std::invalid_argument("MO Decimal256 carrier shape mismatch");
    bool masks = false;
    for (int i = 0; i < 4; ++i) {
      auto child = col.child(i);
      if (child.type().id() != (i ? cudf::type_id::UINT64 : cudf::type_id::INT64) ||
          child.offset() || child.size() != col.size())
        throw std::invalid_argument("MO Decimal256 child shape mismatch");
      masks |= child.nullable();
    }
    if (!masks) return std::move(owner->column);
    owner->bad = std::make_unique<rmm::device_uvector<uint32_t>>(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(owner->bad->data(), 0, sizeof(uint32_t), stream.value()));
    child_validity validity;
    for (int i = 0; i < 4; ++i) {
      validity.masks[i]   = col.child(i).null_mask();
      validity.offsets[i] = col.child(i).offset();
    }
    if (col.size())
      check_children<<<128, block_size, 0, stream.value()>>>(
        view(col, type), validity, col.size(), owner->bad->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    CUDF_CUDA_TRY(cudaMemcpyAsync(&owner->host_bad,
                                  owner->bad->data(),
                                  sizeof(owner->host_bad),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    if (owner->host_bad) throw std::invalid_argument("MO Decimal256 has independent child NULLs");
    auto rows = col.size(), nulls = col.null_count();
    auto contents = owner->column->release();
    for (auto& child : contents.children) {
      auto child_type = child->type();
      auto data       = child->release();
      child           = std::make_unique<cudf::column>(
        child_type, rows, std::move(*data.data), rmm::device_buffer{}, 0);
    }
    owner->column = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::STRUCT},
      rows,
      rmm::device_buffer{},
      contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{},
      nulls,
      std::move(contents.children));
    return std::move(owner->column);
  } catch (...) {
    rethrow_quiescent(owner, stream);
  }
}
void make_group_offsets(cudf::column_view labels,
                        cudf::column_view order,
                        cudf::size_type* offsets,
                        cudf::size_type groups,
                        rmm::cuda_stream_view stream)
{
  group_offsets<<<128, block_size, 0, stream.value()>>>(
    labels.data<cudf::size_type>(), order.data<cudf::size_type>(), labels.size(), offsets, groups);
  CUDF_CUDA_TRY(cudaGetLastError());
}
cudf::column_view canonical_decimal_view(cudf::column_view column,
                                         decimal_type type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (decimal_column_matches(column, type)) return column;
  if (!type.valid() || type.bits != 256 || column.type().id() != cudf::type_id::STRUCT ||
      column.num_children() != 4)
    throw std::invalid_argument("MO exact decimal carrier mismatch");
  child_validity validity;
  std::vector<cudf::column_view> children;
  for (int i = 0; i < 4; ++i) {
    auto child = column.child(i);
    if (child.type().id() != (i ? cudf::type_id::UINT64 : cudf::type_id::INT64) ||
        child.size() < column.offset() + column.size())
      throw std::invalid_argument("MO exact decimal child shape mismatch");
    validity.masks[i]   = child.null_mask();
    validity.offsets[i] = child.offset();
    children.emplace_back(
      child.type(), child.size(), child.head<uint8_t>(), nullptr, 0, child.offset());
  }
  struct owners {
    std::unique_ptr<rmm::device_uvector<uint32_t>> bad;
    uint32_t host_bad{};
  };
  auto owner = std::make_unique<owners>();
  try {
    owner->bad = std::make_unique<rmm::device_uvector<uint32_t>>(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(owner->bad->data(), 0, sizeof(uint32_t), stream.value()));
    if (column.size())
      check_children<<<128, block_size, 0, stream.value()>>>(
        view(column, type), validity, column.size(), owner->bad->data());
    CUDF_CUDA_TRY(cudaGetLastError());
    CUDF_CUDA_TRY(cudaMemcpyAsync(&owner->host_bad,
                                  owner->bad->data(),
                                  sizeof(owner->host_bad),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    if (owner->host_bad)
      throw std::invalid_argument("MO exact decimal contains independent child NULLs");
    return cudf::column_view(column.type(),
                             column.size(),
                             nullptr,
                             column.null_mask(),
                             column.null_count(),
                             column.offset(),
                             std::move(children));
  } catch (...) {
    rethrow_quiescent(owner, stream);
  }
}
}  // namespace sirius::mo_decimal
