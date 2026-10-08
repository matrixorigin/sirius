/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "helper/logical_type.hpp"

#include <duckdb/common/types.hpp>
#include <duckdb/common/types/value.hpp>

#include <optional>

namespace sirius::mo_decimal {
// Private DuckDB carriers preserve the complete MO descriptor. These are not
// registered SQL types and never participate in ordinary decimal inference.
duckdb::LogicalType duckdb_type(logical_type const& type);
std::optional<logical_type> from_duckdb_type(duckdb::LogicalType const& type);
duckdb::Value duckdb_value(coefficient const& value, logical_type const& type);
coefficient from_duckdb_value(duckdb::Value const& value, logical_type const& type);
}  // namespace sirius::mo_decimal
