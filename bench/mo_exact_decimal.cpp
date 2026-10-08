/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/exact_decimal_gpu.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <chrono>
#include <iostream>
#include <vector>

using namespace sirius::mo_decimal;
namespace {
constexpr cudf::size_type rows = 1 << 18;

std::unique_ptr<cudf::column> input(decimal_type type,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto result = make_decimal_column(type, rows, cudf::mask_state::ALL_VALID, stream, mr);
  auto view   = result->mutable_view();
  std::vector<uint64_t> ones(rows, 1);
  if (type.bits == 64) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(view.data<uint64_t>(),
                                  ones.data(),
                                  rows * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
  } else {
    for (int child = 0; child < 3; ++child)
      CUDF_CUDA_TRY(cudaMemsetAsync(
        view.child(child).data<uint64_t>(), 0, rows * sizeof(uint64_t), stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(view.child(3).data<uint64_t>(),
                                  ones.data(),
                                  rows * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
  }
  stream.synchronize();
  return result;
}

template <class Evaluate>
void measure(char const* name, rmm::cuda_stream_view stream, Evaluate evaluate)
{
  // Include allocations, masks, error storage and required stream quiescence.
  // Exclude input upload, warm-up and output destruction for every case.
  for (int repetition = -2; repetition < 7; ++repetition) {
    auto start  = std::chrono::steady_clock::now();
    auto result = evaluate();
    stream.synchronize();
    auto ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (repetition >= 0)
      std::cout << name << ',' << rows << ',' << repetition << ',' << ms << ','
                << rows * 1000.0 / ms << '\n';
  }
}
}  // namespace

int main()
{
  rmm::cuda_stream stream;
  // Match the pooled allocation model of a task resource, and use the same
  // bounded caller-owned pool for cuDF controls and checked operations.
  rmm::mr::pool_memory_resource pool(
    cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}},
    64u << 20,
    128u << 20);
  auto mr = rmm::to_device_async_resource_ref_checked(&pool);
  decimal_type narrow{64, 18, 0}, wide{256, 65, 0};
  auto a = input(narrow, stream.view(), mr);
  auto b = input(narrow, stream.view(), mr);
  auto x = input(wide, stream.view(), mr);
  auto y = input(wide, stream.view(), mr);
  std::cout << "case,rows,repetition,wall_ms,rows_per_second\n";
  for (auto operation : {decimal_op::add, decimal_op::multiply}) {
    auto out = operation == decimal_op::add ? narrow : decimal_type{128, 38, 0};
    auto cudf_op =
      operation == decimal_op::add ? cudf::binary_operator::ADD : cudf::binary_operator::MUL;
    measure(operation == decimal_op::add ? "cudf_add64" : "cudf_widen_multiply64_to128",
            stream.view(),
            [&] {
              auto output_type = cudf::data_type{
                operation == decimal_op::add ? cudf::type_id::DECIMAL64 : cudf::type_id::DECIMAL128,
                0};
              // cuDF arithmetic runs at the operand width. Widen both operands for
              // an equivalent multiply control; changing only the output type wraps.
              std::unique_ptr<cudf::column> left, right;
              if (operation == decimal_op::multiply) {
                left  = cudf::cast(a->view(), output_type, stream.view(), mr);
                right = cudf::cast(b->view(), output_type, stream.view(), mr);
              }
              auto result = cudf::binary_operation(left ? left->view() : a->view(),
                                                   right ? right->view() : b->view(),
                                                   cudf_op,
                                                   output_type,
                                                   stream.view(),
                                                   mr);
              stream.synchronize();
              return result;
            });
    measure(operation == decimal_op::add ? "checked_add64" : "checked_multiply64_to128",
            stream.view(),
            [&] {
              return evaluate_decimal_columns(
                operation, a->view(), narrow, b->view(), narrow, out, nullptr, stream.view(), mr);
            });
  }
  for (auto operation : {decimal_op::add, decimal_op::divide})
    measure(
      operation == decimal_op::add ? "checked_add256" : "checked_divide256", stream.view(), [&] {
        return evaluate_decimal_columns(
          operation, x->view(), wide, y->view(), wide, wide, nullptr, stream.view(), mr);
      });
}
