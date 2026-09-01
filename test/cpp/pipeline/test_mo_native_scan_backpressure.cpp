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

#include <cudf/utilities/default_stream.hpp>

#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/function/table_function.hpp>

#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {

std::unique_ptr<cucascade::memory::memory_space> make_host_space()
{
  cucascade::memory::host_memory_space_config config;
  config.numa_id                    = 0;
  config.memory_capacity            = 128U * 1024U * 1024U;
  config.reservation_limit_fraction = 1.0;
  config.downgrade_trigger_fraction = 1.0;
  config.downgrade_stop_fraction    = 1.0;
  config.block_size                 = 1U * 1024U * 1024U;
  config.pool_size                  = 1;
  config.initial_number_pools       = 0;
  return std::make_unique<cucascade::memory::memory_space>(config);
}

template <typename T>
void append_scalar(std::string& output, T value)
{ output.append(reinterpret_cast<const char*>(&value), sizeof(value)); }

class fixture_native_batch final : public sirius::offload::mo_native_batch {
 public:
  fixture_native_batch(std::uint64_t sequence,
                       std::int64_t value,
                       std::uint32_t rows = 1,
                       bool constant      = false)
    : sequence_(sequence), rows_(rows)
  {
    tae::MOType type{};
    type.oid  = tae::MO_T_int64;
    type.size = sizeof(value);

    sirius::offload::mo_native_column_view column;
    column.vector_class = constant ? 1 : 0;
    column.type         = type;
    column.logical_rows = rows;
    if (constant) {
      append_scalar(encoded_, value);
      column.encoded = encoded_;
      column.data    = encoded_;
      columns_.push_back(column);
      return;
    }

    encoded_.push_back('\0');
    encoded_.append(reinterpret_cast<const char*>(&type), sizeof(type));
    append_scalar(encoded_, rows);
    append_scalar(encoded_, static_cast<std::uint32_t>(sizeof(value) * rows));
    const auto data_offset = encoded_.size();
    for (std::uint32_t row = 0; row < rows; ++row) {
      append_scalar(encoded_, value);
    }
    append_scalar(encoded_, std::uint32_t{0});
    append_scalar(encoded_, std::uint32_t{0});
    encoded_.push_back('\0');
    column.encoded = encoded_;
    column.data    = std::string_view(encoded_).substr(data_offset, sizeof(value) * rows);
    columns_.push_back(column);
  }

  std::uint64_t sequence() const noexcept override { return sequence_; }
  std::uint64_t rows() const noexcept override { return rows_; }
  std::uint64_t payload_bytes() const noexcept override { return encoded_.size(); }
  const std::vector<sirius::offload::mo_native_column_view>& columns() const noexcept override
  { return columns_; }

 private:
  std::uint64_t sequence_;
  std::uint32_t rows_;
  std::string encoded_;
  std::vector<sirius::offload::mo_native_column_view> columns_;
};

class counting_native_source final : public sirius::offload::mo_native_batch_source {
 public:
  counting_native_source()
  {
    batches.push_back(std::make_shared<fixture_native_batch>(1, 11));
    batches.push_back(std::make_shared<fixture_native_batch>(2, 22));
  }

  std::shared_ptr<sirius::offload::mo_native_batch> next_batch() override
  {
    ++next_calls;
    if (next >= batches.size()) { return nullptr; }
    return batches[next++];
  }

  void mark_consumed(std::uint64_t sequence) noexcept override { acknowledged.push_back(sequence); }

  std::vector<std::shared_ptr<sirius::offload::mo_native_batch>> batches;
  std::vector<std::uint64_t> acknowledged;
  std::size_t next       = 0;
  std::size_t next_calls = 0;
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

TEST_CASE("GPU MO scan consumes one frame and releases its claim only after H2D",
          "[mo_native_scan][backpressure]")
{
  auto host_space = make_host_space();

  auto source     = std::make_shared<counting_native_source>();
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get());

  cucascade::shared_data_repository repository;
  auto run_frame = [&](std::uint64_t task_id, std::uint64_t expected_sequence) {
    REQUIRE(scan.get_next_task_hint().has_value());
    REQUIRE(global->try_claim_task());
    REQUIRE_FALSE(global->try_claim_task());

    auto reservation =
      host_space->make_reservation_or_null(sirius::offload::mo_native_scan_reservation_bytes());
    REQUIRE(reservation);
    auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>();
    local->set_reservation(std::move(reservation));
    sirius::op::scan::mo_native_scan_task task(task_id, &repository, std::move(local), global);

    auto output = task.compute_task(cudf::get_default_stream());
    REQUIRE(output);
    REQUIRE(source->next_calls == expected_sequence);
    REQUIRE(source->acknowledged.size() == expected_sequence);
    REQUIRE(source->acknowledged.back() == expected_sequence);
    REQUIRE_FALSE(scan.get_next_task_hint().has_value());
    REQUIRE_FALSE(global->try_claim_task());
    REQUIRE(host_space->get_total_reserved_memory() ==
            sirius::offload::mo_native_scan_reservation_bytes());

    task.publish_output(*output, cudf::get_default_stream());
    REQUIRE_FALSE(scan.get_next_task_hint().has_value());
    auto published = repository.pop_data_batch(cucascade::batch_state::task_created);
    REQUIRE(published);
    auto& host = published->get_data()->cast<sirius::host_tae_representation>();
    host.mark_h2d_complete();
    published.reset();
    REQUIRE(host_space->get_total_reserved_memory() == 0);
    REQUIRE(scan.get_next_task_hint().has_value());
  };

  run_frame(1, 1);
  run_frame(2, 2);

  REQUIRE(global->try_claim_task());
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>();
  sirius::op::scan::mo_native_scan_task eof_task(3, &repository, std::move(local), global);
  REQUIRE_FALSE(eof_task.compute_task(cudf::get_default_stream()));
  REQUIRE(source->next_calls == 3);
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());
  REQUIRE_FALSE(global->try_claim_task());
}

TEST_CASE("GPU MO scan expands one constant frame without cross-frame staging",
          "[mo_native_scan][backpressure]")
{
  auto host_space = make_host_space();

  auto source     = std::make_shared<counting_native_source>();
  source->batches = {std::make_shared<fixture_native_batch>(1, 7, 3, true)};
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get());
  REQUIRE(global->try_claim_task());

  auto reservation =
    host_space->make_reservation_or_null(sirius::offload::mo_native_scan_reservation_bytes());
  REQUIRE(reservation);
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>();
  local->set_reservation(std::move(reservation));
  cucascade::shared_data_repository repository;
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);

  auto output = task.compute_task(cudf::get_default_stream());
  REQUIRE(output);
  auto& pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output);
  REQUIRE(pipelineable.get_data_batches().size() == 1);
  auto batch = pipelineable.get_data_batches().front();
  auto& host = batch->get_data()->cast<sirius::host_tae_representation>();
  REQUIRE(host.get_total_rows() == 3);
  REQUIRE(host.get_host_data()->size() == 58);
  REQUIRE(source->next_calls == 1);
  REQUIRE(source->acknowledged == std::vector<std::uint64_t>{1});
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());

  host.mark_h2d_complete();
  batch.reset();
  output.reset();
  REQUIRE(scan.get_next_task_hint().has_value());
}

TEST_CASE("GPU MO scan rejects a frame outside the cuDF row bound before acknowledgement",
          "[mo_native_scan][backpressure]")
{
  auto host_space = make_host_space();
  auto source     = std::make_shared<counting_native_source>();
  source->batches = {std::make_shared<fixture_native_batch>(
    1, 7, static_cast<std::uint32_t>(std::numeric_limits<cudf::size_type>::max()) + 1U, true)};
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get());
  REQUIRE(global->try_claim_task());

  cucascade::shared_data_repository repository;
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>();
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);
  REQUIRE_THROWS_WITH(task.compute_task(cudf::get_default_stream()),
                      "GPU MO scan received an empty or oversized native batch");
  REQUIRE(source->next_calls == 1);
  REQUIRE(source->acknowledged.empty());
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());
}
