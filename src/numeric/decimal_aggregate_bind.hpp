/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "expression/aggregate_id.hpp"
#include "numeric/decimal_aggregate.hpp"
#include "numeric/decimal_types.hpp"

namespace duckdb {
class DatabaseInstance;
class BoundAggregateExpression;
class Expression;
}  // namespace duckdb
namespace sirius::mo_decimal {
bool is_decimal_aggregate(aggregate_id id) noexcept;
aggregate_op aggregate_operation(aggregate_id id);
aggregate_id aggregate_function(aggregate_op op);
decimal_type aggregate_input_type(logical_type const& input);
void validate_aggregate_signature(aggregate_op op,
                                  logical_type const& input,
                                  logical_type const& output);
void register_aggregate_functions(duckdb::DatabaseInstance& instance);
void validate_bound_aggregate(duckdb::BoundAggregateExpression const& expression);
duckdb::unique_ptr<duckdb::Expression> bound_aggregate(
  aggregate_op op,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children,
  logical_type const& output);
}  // namespace sirius::mo_decimal
