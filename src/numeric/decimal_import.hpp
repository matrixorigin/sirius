/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <duckdb/common/shared_ptr.hpp>

#include <string_view>
namespace duckdb {
class SubstraitExtensionHandler;
}
namespace sirius::mo_decimal {
inline constexpr char extension_uri[] = "urn:matrixone:sirius:exact-decimal:v1";
inline constexpr char literal_type_url[] =
  "type.googleapis.com/matrixone.sirius.numeric.v1.ExactDecimalLiteral";
duckdb::shared_ptr<duckdb::SubstraitExtensionHandler> make_import_handler(std::string_view plan);
}  // namespace sirius::mo_decimal
