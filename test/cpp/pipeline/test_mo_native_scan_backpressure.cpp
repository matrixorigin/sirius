/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "catch.hpp"
#include "helper/type_conversions.hpp"
#include "offload/mo_native_batch.hpp"
#include "op/scan/mo_native_scan_task.hpp"
#include "op/sirius_physical_gpu_mo_scan.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cucascade/data/data_repository.hpp>
#include <duckdb/function/table_function.hpp>

#include <memory>

namespace {

class empty_native_source final : public sirius::offload::mo_native_batch_source {
 public:
  std::shared_ptr<sirius::offload::mo_native_batch> next_batch() override { return nullptr; }
  void mark_consumed(std::uint64_t) noexcept override {}
};

std::unique_ptr<sirius::op::sirius_physical_table_scan> make_stream_table_scan(
  const std::shared_ptr<sirius::offload::mo_native_batch_source>& source)
{
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType::BIGINT};
  duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0)};
  duckdb::vector<duckdb::idx_t> projection_ids{0};
  duckdb::vector<std::string> names{"value"};
  duckdb::vector<duckdb::Value> parameters;
  duckdb::virtual_column_map_t virtual_columns;
  auto filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  duckdb::TableFunction function("mo_stream_scan", {}, nullptr, nullptr);

  return std::make_unique<sirius::op::sirius_physical_table_scan>(
    sirius::from_duckdb_vec(types),
    std::move(function),
    duckdb::make_uniq<sirius::offload::mo_native_scan_bind_data>(source),
    sirius::from_duckdb_vec(types),
    std::move(column_ids),
    std::move(projection_ids),
    std::move(names),
    std::move(filters),
    1,
    duckdb::ExtraOperatorInfo(),
    std::move(parameters),
    std::move(virtual_columns));
}

}  // namespace

TEST_CASE("GPU MO scan advances only after downstream publication demand",
          "[mo_native_scan][backpressure]")
{
  auto manager     = sirius::test::operator_utils::initialize_memory_manager();
  auto host_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE_FALSE(host_spaces.empty());

  auto source     = std::make_shared<empty_native_source>();
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, const_cast<cucascade::memory::memory_space*>(host_spaces.front()));

  REQUIRE(scan.get_next_task_hint().has_value());
  REQUIRE(global->try_claim_task());
  REQUIRE_FALSE(global->try_claim_task());
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());

  cucascade::shared_data_repository repository;
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>();
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);
  sirius::op::pipelineable_operator_data published(
    std::vector<std::shared_ptr<cucascade::data_batch>>{});
  task.publish_output(published, cudf::get_default_stream());

  REQUIRE(scan.get_next_task_hint().has_value());
  REQUIRE(global->try_claim_task());
  global->finish_eof();
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());
  REQUIRE_FALSE(global->try_claim_task());
}
