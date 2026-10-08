/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "numeric/decimal_types.hpp"

#include <duckdb/planner/expression.hpp>

#include <string_view>
namespace duckdb {
class LogicalOperator;
class ClientContext;
}  // namespace duckdb
namespace sirius::mo_decimal {
std::string normalize_exact_substrait(std::string_view bytes,
                                      std::size_t schema_bytes_limit = 16u << 20);
void restore_exact_bound_types(duckdb::unique_ptr<duckdb::Expression>& expression);
void rewrite_exact_comparisons(duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                               duckdb::ClientContext& context);
}  // namespace sirius::mo_decimal
