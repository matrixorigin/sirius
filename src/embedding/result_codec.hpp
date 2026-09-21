/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "embedding/result.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
namespace sirius::embedding {
struct result_slice_layout {
  uint32_t rows{};
  std::size_t bytes{};
};
result_slice_layout size_native_result(cudf::table_view table,
                                       uint32_t begin,
                                       std::span<const owned_column> schema,
                                       result_batch& scratch,
                                       rmm::cuda_stream_view stream,
                                       std::size_t target);
void encode_native_result(cudf::table_view table,
                          uint32_t begin,
                          std::span<const owned_column> schema,
                          result_slice_layout const& layout,
                          result_batch& output,
                          result_batch& scratch,
                          rmm::cuda_stream_view stream);
}  // namespace sirius::embedding
