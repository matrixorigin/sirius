/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "numeric/exact_decimal.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>

namespace sirius::mo_decimal {
// Decimal64/128 retain cuDF's existing carriers. Decimal256 uses the numeric
// ordering [signed high64, unsigned next64, unsigned next64, unsigned low64].
std::unique_ptr<cudf::column> make_decimal_column(decimal_type type,
                                                  cudf::size_type rows,
                                                  cudf::mask_state mask,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);
bool decimal_column_matches(cudf::column_view const& column, decimal_type type);

struct decimal_column_result {
  std::unique_ptr<cudf::column> values;
  rmm::device_uvector<uint8_t> errors;
};
std::unique_ptr<cudf::column> make_decimal_literal(decimal_type type,
                                                   coefficient value,
                                                   bool valid,
                                                   cudf::size_type rows,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr);
decimal_error column_error(decimal_column_result const& result,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);
// Caller retains operands/mask and the returned buffers until its task stream
// is quiescent. All allocations use that stream's reservation-aware resource.
decimal_column_result evaluate_decimal_columns(decimal_op op,
                                               cudf::column_view left,
                                               decimal_type left_type,
                                               cudf::column_view right,
                                               decimal_type right_type,
                                               decimal_type output_type,
                                               bool const* active_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);
}  // namespace sirius::mo_decimal
