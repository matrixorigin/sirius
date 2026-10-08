/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "expression/ast/node.hpp"
#include "numeric/decimal_aggregate_bind.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace sirius::mo_decimal {
struct aggregate_slot {
  aggregate_id id;
  int input;
  logical_type input_type, output_type;
  std::vector<int> struct_inputs;
  bool distinct{};
  std::size_t state_index{};
  bool exact() const { return is_decimal_aggregate(id); }
  bool avg() const { return id == aggregate_id::avg || id == aggregate_id::mo_decimal_avg; }
  uint16_t state_bits() const
  {
    return exact()
             ? static_cast<uint16_t>(std::max<uint16_t>(128, output_type.mo_decimal_type().bits))
             : 0;
  }
};
struct aggregate_layout {
  std::vector<int> group_indices;
  duckdb::vector<logical_type> group_types;
  std::vector<aggregate_slot> slots;
  duckdb::vector<logical_type> local_types() const;
};
// Returns null when the unchanged ordinary aggregate implementation applies.
std::shared_ptr<aggregate_layout const> make_aggregate_layout(
  duckdb::vector<std::unique_ptr<ast::node>> const& groups,
  duckdb::vector<std::unique_ptr<ast::node>> const& aggregates);
std::unique_ptr<cudf::table> local_aggregate_table(cudf::table_view input,
                                                   aggregate_layout const& layout,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr);
std::unique_ptr<cudf::table> merge_aggregate_tables(std::vector<cudf::table_view> const& inputs,
                                                    aggregate_layout const& layout,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr);
}  // namespace sirius::mo_decimal
