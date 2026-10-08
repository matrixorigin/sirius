/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "expression/function_id.hpp"
#include "numeric/decimal_types.hpp"

#include <duckdb/function/function.hpp>
#include <duckdb/planner/expression.hpp>

namespace duckdb {
class DatabaseInstance;
class BoundFunctionExpression;
}  // namespace duckdb
namespace sirius::ast {
struct node;
}

namespace sirius::mo_decimal {
bool is_decimal_function(function_id id) noexcept;
decimal_op operation(function_id id);
function_id function(decimal_op op);
bool contains_exact_expression(ast::node const& expression);
void validate_signature(decimal_op op,
                        std::vector<logical_type> const& inputs,
                        logical_type const& output);
void register_scalar_functions(duckdb::DatabaseInstance& instance);
void validate_bound_scalar(duckdb::BoundFunctionExpression const& expression);
duckdb::unique_ptr<duckdb::Expression> bound_scalar(
  decimal_op op,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children,
  logical_type const& output);
}  // namespace sirius::mo_decimal
