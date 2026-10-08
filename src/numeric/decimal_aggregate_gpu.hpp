/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "numeric/decimal_aggregate.hpp"
#include "numeric/exact_decimal_gpu.hpp"

namespace sirius::mo_decimal {
struct aggregate_columns {
  std::unique_ptr<cudf::column> values;
  std::unique_ptr<cudf::column> counts;
};

// Offsets cover dense groups in the permutation (or original rows when order
// is empty). Empty counts means initial rows; otherwise these are raw states.
aggregate_columns reduce_aggregate_columns(aggregate_op op,
                                           cudf::column_view values,
                                           decimal_type input,
                                           uint16_t state_bits,
                                           cudf::column_view counts,
                                           cudf::column_view order,
                                           cudf::size_type const* offsets,
                                           cudf::size_type groups,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::column> finalize_aggregate_columns(aggregate_op op,
                                                         cudf::column_view values,
                                                         cudf::column_view counts,
                                                         decimal_type input,
                                                         decimal_type output,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr);
// Build the same scale-independent key for equality partitioning and probing.
std::unique_ptr<cudf::column> make_equality_key(cudf::column_view values,
                                                decimal_type type,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);
// cuDF gather/selection can superimpose parent validity on STRUCT children.
// Verify there are no independent child NULLs before removing those masks.
std::unique_ptr<cudf::column> restore_decimal_validity(std::unique_ptr<cudf::column> column,
                                                       decimal_type type,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr);
cudf::column_view canonical_decimal_view(cudf::column_view column,
                                         decimal_type type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);
void make_group_offsets(cudf::column_view labels,
                        cudf::column_view order,
                        cudf::size_type* offsets,
                        cudf::size_type groups,
                        rmm::cuda_stream_view stream);
}  // namespace sirius::mo_decimal
