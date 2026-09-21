/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/plan_bindings.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/main/client_context.hpp"

namespace sirius::embedding {
void embedded_bind_catalog::declare(uint64_t id, embedded_binding binding)
{
  std::lock_guard lock(mutex_);
  entries_.insert_or_assign(id, std::move(binding));
}
embedded_binding embedded_bind_catalog::get(uint64_t id) const
{
  std::lock_guard lock(mutex_);
  auto found = entries_.find(id);
  if (found == entries_.end()) throw std::runtime_error("undeclared embedded read binding");
  return found->second;
}
void embedded_bind_catalog::clear()
{
  std::lock_guard lock(mutex_);
  entries_.clear();
}
duckdb::shared_ptr<embedded_bind_catalog> embedded_catalog_for(duckdb::ClientContext& context)
{
  auto catalog = context.registered_state->Get<embedded_bind_catalog>(embedded_bind_catalog::key);
  if (!catalog) throw std::runtime_error("embedded read catalog is not installed");
  return catalog;
}
namespace {
uint64_t read_id(duckdb::TableFunctionBindInput& input)
{
  if (input.inputs.size() != 1 || input.inputs[0].IsNull())
    throw std::runtime_error("embedded read requires one binding id");
  auto id = input.inputs[0].GetValue<int64_t>();
  if (id <= 0) throw std::runtime_error("invalid embedded read binding id");
  return static_cast<uint64_t>(id);
}
duckdb::unique_ptr<duckdb::FunctionData> bind_mo(duckdb::ClientContext& context,
                                                 duckdb::TableFunctionBindInput& input,
                                                 duckdb::vector<duckdb::LogicalType>& types,
                                                 duckdb::vector<std::string>& names)
{
  auto binding = embedded_catalog_for(context)->get(read_id(input));
  if (!binding.input || binding.tae) throw std::runtime_error("binding is not an MO read");
  types       = {binding.types.begin(), binding.types.end()};
  names       = {binding.names.begin(), binding.names.end()};
  auto data   = duckdb::make_uniq<embedded_mo_bind_data>();
  data->input = std::move(binding.input);
  return data;
}
duckdb::unique_ptr<duckdb::FunctionData> bind_tae(duckdb::ClientContext& context,
                                                  duckdb::TableFunctionBindInput& input,
                                                  duckdb::vector<duckdb::LogicalType>& types,
                                                  duckdb::vector<std::string>& names)
{
  auto binding = embedded_catalog_for(context)->get(read_id(input));
  if (!binding.tae || !binding.tae_host_budget || binding.input)
    throw std::runtime_error("binding is not a complete TAE read");
  types             = {binding.types.begin(), binding.types.end()};
  names             = {binding.names.begin(), binding.names.end()};
  auto data         = duckdb::make_uniq<embedded_tae_bind_data>();
  data->manifest    = std::move(binding.tae);
  data->host_budget = std::move(binding.tae_host_budget);
  return data;
}
void never_execute(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&)
{
  throw std::runtime_error("embedded read marker cannot execute in DuckDB");
}
}  // namespace
void register_embedded_read_functions(duckdb::DatabaseInstance& instance)
{
  auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = duckdb::Catalog::GetSystemCatalog(instance);
  for (auto fn :
       {duckdb::TableFunction(
          embedded_mo_function, {duckdb::LogicalType::BIGINT}, never_execute, bind_mo),
        duckdb::TableFunction(
          embedded_tae_function, {duckdb::LogicalType::BIGINT}, never_execute, bind_tae)}) {
    fn.projection_pushdown = true;
    fn.filter_pushdown     = true;
    duckdb::CreateTableFunctionInfo info(fn);
    info.on_conflict = duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
    catalog.CreateTableFunction(transaction, info);
  }
}
}  // namespace sirius::embedding
