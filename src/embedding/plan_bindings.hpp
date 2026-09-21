/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "embedding/input.hpp"
#include "tae_scanner.hpp"

#include <map>
#include <mutex>

namespace sirius::embedding {
inline constexpr const char* embedded_mo_function  = "sirius_embedded_mo_read";
inline constexpr const char* embedded_tae_function = "sirius_embedded_tae_read";

struct embedded_mo_bind_data final : duckdb::TableFunctionData {
  std::shared_ptr<native_input> input;
  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    auto copy   = duckdb::make_uniq<embedded_mo_bind_data>();
    copy->input = input;
    return copy;
  }
};

// Query resources belong to Sirius, not the reusable storage manifest parser.
// The carrier type identifies embedded mode even without optional sort metadata.
struct embedded_tae_bind_data final : duckdb::TableFunctionData {
  std::shared_ptr<const tae::TAEScanBindData> manifest;
  std::shared_ptr<buffer_budget> host_budget;
  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    auto copy         = duckdb::make_uniq<embedded_tae_bind_data>();
    copy->manifest    = manifest;
    copy->host_budget = host_budget;
    return copy;
  }
};

struct embedded_binding {
  std::vector<std::string> names;
  std::vector<duckdb::LogicalType> types;
  std::shared_ptr<native_input> input;
  std::shared_ptr<const tae::TAEScanBindData> tae;
  std::shared_ptr<buffer_budget> tae_host_budget;
};

class embedded_bind_catalog final : public duckdb::ClientContextState {
 public:
  static constexpr const char* key = "sirius_embedded_bind_catalog";
  void declare(uint64_t id, embedded_binding binding);
  embedded_binding get(uint64_t id) const;
  void clear();

 private:
  mutable std::mutex mutex_;
  std::map<uint64_t, embedded_binding> entries_;
};

void register_embedded_read_functions(duckdb::DatabaseInstance&);
duckdb::shared_ptr<embedded_bind_catalog> embedded_catalog_for(duckdb::ClientContext&);
}  // namespace sirius::embedding
