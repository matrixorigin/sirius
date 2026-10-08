/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "numeric/decimal_types.hpp"

#include <duckdb/common/shared_ptr.hpp>

#include <string_view>
namespace duckdb {
class SubstraitExtensionHandler;
}
namespace substrait {
class Type;
}
namespace sirius::mo_decimal {
inline constexpr char extension_uri[] = "urn:matrixone:sirius:exact-decimal:v1";
inline constexpr char literal_type_url[] =
  "type.googleapis.com/matrixone.sirius.numeric.v1.ExactDecimalLiteral";
duckdb::shared_ptr<duckdb::SubstraitExtensionHandler> make_import_handler(std::string_view plan);
bool uses_exact_decimal(std::string_view plan);
logical_type import_result_type(substrait::Type const& type, std::vector<uint32_t> const& anchors);
}  // namespace sirius::mo_decimal
