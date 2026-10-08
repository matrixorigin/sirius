/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/exact_decimal_gpu.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <stdexcept>
#include <vector>

namespace sirius::mo_decimal {
namespace {
__global__ void broadcast_literal(coefficient value,
                                  uint32_t bytes,
                                  cudf::size_type rows,
                                  uint8_t* data,
                                  uint64_t* high,
                                  uint64_t* mid_high,
                                  uint64_t* mid_low,
                                  uint64_t* low)
{
  for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    if (bytes != 32)
      store_coefficient(value, data + row * bytes, bytes);
    else {
      uint64_t* limbs[]{low, mid_low, mid_high, high};
      for (int i = 0; i < 4; ++i)
        limbs[i][row] = uint64_t(value.words[2 * i]) | uint64_t(value.words[2 * i + 1]) << 32;
    }
  }
}
struct operand {
  uint8_t const* data{};
  uint64_t const* limbs[4]{};
  cudf::bitmask_type const* mask{};
  cudf::size_type offset{};
  uint32_t bytes{};
  __device__ bool valid(cudf::size_type row) const
  {
    auto bit = offset + row;
    return !mask || ((mask[bit / 32] >> (bit % 32)) & 1);
  }
  __device__ coefficient value(cudf::size_type row) const
  {
    if (bytes != 32) return load_coefficient(data + uint64_t(row + offset) * bytes, bytes);
    coefficient result;
    for (int i = 0; i < 4; ++i) {
      auto value              = limbs[3 - i][row + offset];
      result.words[2 * i]     = value;
      result.words[2 * i + 1] = value >> 32;
    }
    return result;
  }
};
operand view(cudf::column_view const& column, decimal_type type)
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
__global__ void evaluate_fast64(decimal_op op,
                                operand left,
                                operand right,
                                cudf::size_type rows,
                                uint64_t bound,
                                uint8_t* output,
                                cudf::bitmask_type* validity,
                                uint8_t* errors,
                                uint32_t* valid_count,
                                bool const* active)
{
  for (int64_t base = blockIdx.x * blockDim.x; base < rows; base += blockDim.x * gridDim.x) {
    auto row = base + threadIdx.x;
    decimal_result result;
    if (row < rows) {
      bool enabled = !active || active[row];
      bool lv = enabled && left.valid(row), rv = enabled && right.valid(row);
      uint64_t a  = lv ? reinterpret_cast<uint64_t const*>(left.data)[row + left.offset] : 0;
      uint64_t b  = rv ? reinterpret_cast<uint64_t const*>(right.data)[row + right.offset] : 0;
      result      = evaluate_same_scale64(op, a, b, bound, lv, rv, enabled);
      errors[row] = static_cast<uint8_t>(result.error);
      if (op >= decimal_op::equal)
        output[row] = result.value.words[0];
      else
        reinterpret_cast<uint64_t*>(output)[row] =
          uint64_t(result.value.words[0]) | (uint64_t(result.value.words[1]) << 32);
    }
    auto bits = __ballot_sync(0xffffffff, row < rows && result.valid);
    if ((threadIdx.x % 32) == 0 && row < rows) {
      validity[row / 32] = bits;
      atomicAdd(valid_count, static_cast<uint32_t>(__popc(bits)));
    }
  }
}
__global__ void evaluate(decimal_op op,
                         operand left,
                         decimal_type left_type,
                         operand right,
                         decimal_type right_type,
                         decimal_type output_type,
                         cudf::size_type rows,
                         uint8_t* output,
                         uint64_t* high,
                         uint64_t* mid_high,
                         uint64_t* mid_low,
                         uint64_t* low,
                         cudf::bitmask_type* validity,
                         uint8_t* errors,
                         bool const* active)
{
  for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    bool enabled = !active || active[row];
    bool lv = enabled && left.valid(row), rv = enabled && right.valid(row);
    coefficient a{}, b{};
    if (lv) a = left.value(row);
    if (rv) b = right.value(row);
    auto result = evaluate_decimal(op, a, left_type, lv, b, right_type, rv, output_type, enabled);
    errors[row] = static_cast<uint8_t>(result.error);
    if (result.valid) atomicOr(validity + row / 32, uint32_t(1) << (row % 32));
    if (op >= decimal_op::equal)
      output[row] = result.value.words[0];
    else if (op == decimal_op::integer_divide)
      store_coefficient(result.value, output + uint64_t(row) * 8, 8);
    else if (output_type.bits != 256)
      store_coefficient(
        result.value, output + uint64_t(row) * output_type.bytes(), output_type.bytes());
    else {
      uint64_t* limbs[]{low, mid_low, mid_high, high};
      for (int i = 0; i < 4; ++i)
        limbs[i][row] =
          uint64_t(result.value.words[i * 2]) | (uint64_t(result.value.words[i * 2 + 1]) << 32);
    }
  }
}
}  // namespace

std::unique_ptr<cudf::column> make_decimal_literal(decimal_type type,
                                                   coefficient value,
                                                   bool valid,
                                                   cudf::size_type rows,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (!type.valid() || rows < 0)
    throw std::invalid_argument("invalid MO exact-decimal literal descriptor");
  if (valid && evaluate_decimal(decimal_op::cast, value, type, true, {}, type, true, type).error !=
                 decimal_error::none)
    throw std::invalid_argument("MO exact-decimal literal exceeds declared precision");
  auto owner = std::make_unique<std::unique_ptr<cudf::column>>();
  try {
    *owner = make_decimal_column(
      type, rows, valid ? cudf::mask_state::ALL_VALID : cudf::mask_state::ALL_NULL, stream, mr);
    if (rows) {
      auto view = (*owner)->mutable_view();
      broadcast_literal<<<128, 256, 0, stream.value()>>>(
        valid ? value : coefficient{},
        type.bytes(),
        rows,
        type.bits == 256 ? nullptr : view.data<uint8_t>(),
        type.bits == 256 ? view.child(0).data<uint64_t>() : nullptr,
        type.bits == 256 ? view.child(1).data<uint64_t>() : nullptr,
        type.bits == 256 ? view.child(2).data<uint64_t>() : nullptr,
        type.bits == 256 ? view.child(3).data<uint64_t>() : nullptr);
      CUDF_CUDA_TRY(cudaGetLastError());
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    }
    return std::move(*owner);
  } catch (...) {
    auto error = std::current_exception();
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error("MO decimal literal could not prove quiescence");
    }
    std::rethrow_exception(error);
  }
}
decimal_error column_error(decimal_column_result const& result,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  if (!result.errors.size()) return decimal_error::none;
  cudf::column_view errors(
    cudf::data_type{cudf::type_id::UINT8}, result.errors.size(), result.errors.data(), nullptr, 0);
  auto owner = std::make_unique<std::unique_ptr<cudf::scalar>>();
  try {
    *owner    = cudf::reduce(errors,
                          *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
                          cudf::data_type{cudf::type_id::UINT8},
                          stream,
                          mr);
    auto code = static_cast<cudf::numeric_scalar<uint8_t>&>(**owner).value(stream);
    return static_cast<decimal_error>(code);
  } catch (...) {
    auto error = std::current_exception();
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error(
        "MO decimal error reduction could not prove quiescence");
    }
    std::rethrow_exception(error);
  }
}

std::unique_ptr<cudf::column> make_decimal_column(decimal_type type,
                                                  cudf::size_type rows,
                                                  cudf::mask_state mask,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  if (!type.valid() || rows < 0 || mask == cudf::mask_state::UNINITIALIZED)
    throw std::invalid_argument("invalid MO exact decimal descriptor or mask state");
  if (type.bits != 256)
    return cudf::make_fixed_width_column(
      cudf::data_type{type.bits == 64 ? cudf::type_id::DECIMAL64 : cudf::type_id::DECIMAL128,
                      -static_cast<int32_t>(type.scale)},
      rows,
      mask,
      stream,
      mr);
  std::vector<std::unique_ptr<cudf::column>> children;
  for (int i = 0; i < 4; ++i)
    children.push_back(cudf::make_numeric_column(
      cudf::data_type{i == 0 ? cudf::type_id::INT64 : cudf::type_id::UINT64},
      rows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr));
  auto validity = cudf::create_null_mask(rows, mask, stream, mr);
  auto nulls    = mask == cudf::mask_state::ALL_NULL ? rows : 0;
  return cudf::create_structs_hierarchy(
    rows, std::move(children), nulls, std::move(validity), stream, mr);
}
bool decimal_column_matches(cudf::column_view const& column, decimal_type type)
{
  if (!type.valid()) return false;
  if (type.bits != 256)
    return column.type() ==
           cudf::data_type{type.bits == 64 ? cudf::type_id::DECIMAL64 : cudf::type_id::DECIMAL128,
                           -static_cast<int32_t>(type.scale)};
  if (column.type().id() != cudf::type_id::STRUCT || column.num_children() != 4) return false;
  for (int i = 0; i < 4; ++i) {
    auto child = column.child(i);
    if (child.type().id() != (i == 0 ? cudf::type_id::INT64 : cudf::type_id::UINT64) ||
        child.nullable() || child.size() < column.offset() + column.size())
      return false;
  }
  return true;
}
decimal_column_result evaluate_decimal_columns(decimal_op op,
                                               cudf::column_view left,
                                               decimal_type left_type,
                                               cudf::column_view right,
                                               decimal_type right_type,
                                               decimal_type output_type,
                                               bool const* active,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (!decimal_column_matches(left, left_type) || !decimal_column_matches(right, right_type) ||
      !output_type.valid() || left.size() != right.size() ||
      static_cast<uint8_t>(op) > static_cast<uint8_t>(decimal_op::greater_equal))
    throw std::invalid_argument("invalid MO exact decimal operands");
  auto rows = left.size();
  // These signatures contain the entire signed 64-bit physical input domain,
  // not merely sampled values. No rounding or physical/precision overflow is
  // possible, so retain the existing cuDF implementation rather than emulate it.
  bool safe_add = (op == decimal_op::add || op == decimal_op::subtract) && left_type.bits == 64 &&
                  right_type.bits == 64 && output_type.bits == 128 && output_type.precision >= 20 &&
                  left_type.scale == right_type.scale && left_type.scale == output_type.scale;
  bool safe_multiply = op == decimal_op::multiply && left_type.bits == 64 &&
                       right_type.bits == 64 && output_type.bits == 128 &&
                       output_type.precision == 38 &&
                       output_type.scale == left_type.scale + right_type.scale;
  if (!active && (safe_add || safe_multiply)) {
    struct widened_owner {
      decimal_column_result result;
      std::unique_ptr<cudf::column> left, right;
    };
    auto owner = std::make_unique<widened_owner>(
      widened_owner{{nullptr, rmm::device_uvector<uint8_t>(rows, stream, mr)}, nullptr, nullptr});
    try {
      auto operation = op == decimal_op::add        ? cudf::binary_operator::ADD
                       : op == decimal_op::subtract ? cudf::binary_operator::SUB
                                                    : cudf::binary_operator::MUL;
      owner->left    = cudf::cast(
        left,
        cudf::data_type{cudf::type_id::DECIMAL128, -static_cast<int32_t>(left_type.scale)},
        stream,
        mr);
      owner->right = cudf::cast(
        right,
        cudf::data_type{cudf::type_id::DECIMAL128, -static_cast<int32_t>(right_type.scale)},
        stream,
        mr);
      owner->result.values = cudf::binary_operation(
        owner->left->view(),
        owner->right->view(),
        operation,
        cudf::data_type{cudf::type_id::DECIMAL128, -static_cast<int32_t>(output_type.scale)},
        stream,
        mr);
      if (rows)
        CUDF_CUDA_TRY(cudaMemsetAsync(owner->result.errors.data(), 0, rows, stream.value()));
      // cuDF may enqueue readers of the widened temporaries. Do not let their
      // owner disappear before those reads and result writes are quiescent.
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
      return std::move(owner->result);
    } catch (...) {
      auto error = std::current_exception();
      if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
        (void)owner.release();
        throw pipeline::gpu_stream_quiescence_error(
          "exact decimal fast path could not prove quiescence");
      }
      std::rethrow_exception(error);
    }
  }
  std::unique_ptr<cudf::column> output;
  if (op >= decimal_op::equal || op == decimal_op::integer_divide)
    output = cudf::make_numeric_column(
      cudf::data_type{op >= decimal_op::equal ? cudf::type_id::BOOL8 : cudf::type_id::INT64},
      rows,
      cudf::mask_state::ALL_NULL,
      stream,
      mr);
  else
    output = make_decimal_column(output_type, rows, cudf::mask_state::ALL_NULL, stream, mr);
  struct kernel_owner {
    decimal_column_result result;
    rmm::device_uvector<uint32_t> count;
  };
  auto owner = std::make_unique<kernel_owner>(
    kernel_owner{{std::move(output), rmm::device_uvector<uint8_t>(rows, stream, mr)},
                 rmm::device_uvector<uint32_t>(1, stream, mr)});
  try {
    if (rows) {
      auto ov     = owner->result.values->mutable_view();
      bool wide   = ov.type().id() == cudf::type_id::STRUCT;
      bool fast64 = left_type.bits == 64 && right_type.bits == 64 && output_type.bits == 64 &&
                    left_type.scale == right_type.scale && left_type.scale == output_type.scale &&
                    (op == decimal_op::add || op == decimal_op::subtract ||
                     op == decimal_op::negate || op == decimal_op::cast || op >= decimal_op::equal);
      if (fast64) {
        CUDF_CUDA_TRY(cudaMemsetAsync(owner->count.data(), 0, sizeof(uint32_t), stream.value()));
        uint64_t bound = 1;
        for (int i = 0; i < output_type.precision; ++i)
          bound *= 10;
        evaluate_fast64<<<128, 256, 0, stream.value()>>>(op,
                                                         view(left, left_type),
                                                         view(right, right_type),
                                                         rows,
                                                         bound,
                                                         ov.data<uint8_t>(),
                                                         ov.null_mask(),
                                                         owner->result.errors.data(),
                                                         owner->count.data(),
                                                         active);
      } else
        evaluate<<<128, 256, 0, stream.value()>>>(op,
                                                  view(left, left_type),
                                                  left_type,
                                                  view(right, right_type),
                                                  right_type,
                                                  output_type,
                                                  rows,
                                                  wide ? nullptr : ov.data<uint8_t>(),
                                                  wide ? ov.child(0).data<uint64_t>() : nullptr,
                                                  wide ? ov.child(1).data<uint64_t>() : nullptr,
                                                  wide ? ov.child(2).data<uint64_t>() : nullptr,
                                                  wide ? ov.child(3).data<uint64_t>() : nullptr,
                                                  ov.null_mask(),
                                                  owner->result.errors.data(),
                                                  active);
      CUDF_CUDA_TRY(cudaGetLastError());
      if (fast64) {
        uint32_t count;
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          &count, owner->count.data(), sizeof(count), cudaMemcpyDeviceToHost, stream.value()));
        CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
        owner->result.values->set_null_count(rows - count);
      } else
        owner->result.values->set_null_count(cudf::null_count(ov.null_mask(), 0, rows, stream));
    }
    return std::move(owner->result);
  } catch (...) {
    auto failure = std::current_exception();
    if (cudaStreamSynchronize(stream.value()) != cudaSuccess) {
      (void)owner.release();
      throw pipeline::gpu_stream_quiescence_error(
        "exact decimal kernel could not prove quiescence");
    }
    std::rethrow_exception(failure);
  }
}
}  // namespace sirius::mo_decimal
