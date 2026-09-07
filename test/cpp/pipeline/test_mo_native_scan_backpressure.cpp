/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "catch.hpp"
#include "cuda/tae/tae_decode_kernels.hpp"
#include "helper/type_conversions.hpp"
#include "offload/mo_native_batch.hpp"
#include "op/scan/mo_native_scan_task.hpp"
#include "op/sirius_physical_gpu_mo_scan.hpp"
#include "op/sirius_physical_table_scan.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/function/table_function.hpp>

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

std::unique_ptr<cucascade::memory::memory_space> make_host_space(std::size_t block_size = 1U *
                                                                                          1024U *
                                                                                          1024U)
{
  cucascade::memory::host_memory_space_config config;
  config.numa_id                    = 0;
  config.memory_capacity            = 128U * 1024U * 1024U;
  config.reservation_limit_fraction = 1.0;
  config.downgrade_trigger_fraction = 1.0;
  config.downgrade_stop_fraction    = 1.0;
  config.block_size                 = block_size;
  config.pool_size                  = 1;
  config.initial_number_pools       = 0;
  return std::make_unique<cucascade::memory::memory_space>(config);
}

template <typename T>
void append_scalar(std::string& output, T value)
{
  output.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

class fixture_native_batch final : public sirius::offload::mo_native_batch {
 public:
  fixture_native_batch(std::uint64_t sequence,
                       std::int64_t value,
                       std::uint32_t rows = 32,
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
  {
    return columns_;
  }

  static std::shared_ptr<fixture_native_batch> null_string()
  {
    auto result      = std::make_shared<fixture_native_batch>(1, 0, 1);
    auto& column     = result->columns_.front();
    column.type.oid  = tae::MO_T_varchar;
    column.type.size = -24;
    auto& encoded    = result->encoded_;
    encoded.clear();
    encoded.push_back(0);
    encoded.append(reinterpret_cast<const char*>(&column.type), sizeof(column.type));
    append_scalar(encoded, std::uint32_t{1});
    append_scalar(encoded, std::uint32_t{tae::VARLENA_SIZE});
    const auto data_offset = encoded.size();
    encoded.append(tae::VARLENA_SIZE, '\xff');
    append_scalar(encoded, std::uint32_t{0});
    append_scalar(encoded, std::uint32_t{32});
    append_scalar(encoded, std::uint64_t{1});
    append_scalar(encoded, std::uint64_t{1});
    append_scalar(encoded, std::uint64_t{8});
    const auto null_offset = encoded.size();
    append_scalar(encoded, std::uint64_t{1});
    encoded.push_back(0);
    column.encoded    = encoded;
    column.data       = std::string_view(encoded).substr(data_offset, tae::VARLENA_SIZE);
    column.null_words = std::string_view(encoded).substr(null_offset, 8);
    column.null_count = 1;
    return result;
  }

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

  sirius::offload::mo_native_batch_source_result next_batch() override
  {
    ++next_calls;
    if (not_needed || acknowledged.size() >= not_needed_after) {
      return {sirius::offload::mo_native_batch_source_status::NOT_NEEDED, nullptr};
    }
    if (next >= batches.size()) {
      return {sirius::offload::mo_native_batch_source_status::END_OF_STREAM, nullptr};
    }
    return {sirius::offload::mo_native_batch_source_status::BATCH, batches[next]};
  }

  bool mark_consumed(std::uint64_t sequence) override
  {
    if (not_needed) { return false; }
    if (next >= batches.size() || batches[next]->sequence() != sequence) {
      throw std::runtime_error("unexpected native input acknowledgement");
    }
    acknowledged.push_back(sequence);
    ++next;
    return true;
  }

  std::vector<std::shared_ptr<sirius::offload::mo_native_batch>> batches;
  std::vector<std::uint64_t> acknowledged;
  std::size_t next             = 0;
  std::size_t next_calls       = 0;
  bool not_needed              = false;
  std::size_t not_needed_after = std::numeric_limits<std::size_t>::max();
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

TEST_CASE("pooled native host buffers copy across blocks and release their reservation",
          "[mo_native_scan][backpressure][host_buffer]")
{
  auto space = make_host_space(1024);
  auto* pool = space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  REQUIRE(pool);
  auto stream = cudf::get_default_stream();
  std::vector<std::uint8_t> expected(2500);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = (i * 17) % 251;
  }
  std::size_t pool_blocks = 0;
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto reservation = space->make_reservation_or_null(expected.size());
    REQUIRE(reservation);
    {
      sirius::pinned_host_buffer buffer(expected.size(), *pool, std::move(reservation));
      REQUIRE(buffer.is_pinned());
      REQUIRE_THROWS(buffer.data());
      buffer.copy_from(0, expected.data(), 1021);
      buffer.copy_from(1021, expected.data() + 1021, expected.size() - 1021);
      REQUIRE_THROWS(buffer.copy_from(expected.size(), expected.data(), 1));
      buffer.set_logical_size(expected.size() - 7);
      sirius::pinned_host_buffer moved;
      moved = std::move(buffer);
      REQUIRE(moved.capacity() == expected.size());
      REQUIRE(space->get_total_reserved_memory() >= expected.size());

      rmm::device_buffer device(expected.size(), stream);
      CUDF_CUDA_TRY(cudaMemsetAsync(device.data(), 0xA5, device.size(), stream.value()));
      CUDF_CUDA_TRY(moved.copy_to_device(device.data(), stream));
      std::vector<std::uint8_t> actual(expected.size());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        actual.data(), device.data(), actual.size(), cudaMemcpyDeviceToHost, stream.value()));
      stream.synchronize();
      REQUIRE(std::equal(expected.begin(), expected.end() - 7, actual.begin()));
      REQUIRE(
        std::all_of(actual.end() - 7, actual.end(), [](auto value) { return value == 0xA5; }));
      if (iteration == 0) { pool_blocks = pool->get_total_blocks(); }
      REQUIRE(pool->get_total_blocks() == pool_blocks);
    }
    REQUIRE(space->get_total_reserved_memory() == 0);
    REQUIRE(pool->get_total_allocated_bytes() == 0);
    REQUIRE(pool->get_free_blocks() == pool_blocks);
  }
}

TEST_CASE("GPU MO scan batches frames and releases its claim only after H2D",
          "[mo_native_scan][backpressure][batching]")
{
  auto host_space = make_host_space();

  auto source     = std::make_shared<counting_native_source>();
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  const auto frame_bytes = source->batches.front()->columns().front().encoded.size();
  const auto capacity    = frame_bytes * 3;
  auto global            = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get(), frame_bytes * 2, capacity);

  cucascade::shared_data_repository repository;
  REQUIRE(scan.get_next_task_hint().has_value());
  REQUIRE(global->try_claim_task());
  REQUIRE_FALSE(global->try_claim_task());

  auto reservation = host_space->make_reservation_or_null(capacity);
  REQUIRE(reservation);
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(capacity);
  local->set_reservation(std::move(reservation));
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);

  auto output = task.compute_task(cudf::get_default_stream());
  REQUIRE(output);
  REQUIRE(source->next_calls == 2);
  REQUIRE(source->acknowledged == std::vector<std::uint64_t>{1, 2});
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());
  REQUIRE_FALSE(global->try_claim_task());
  REQUIRE(host_space->get_total_reserved_memory() >= capacity);

  auto& pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output);
  REQUIRE(pipelineable.get_data_batches().size() == 1);
  auto batch = pipelineable.get_data_batches().front();
  auto& host = batch->get_data()->cast<sirius::host_tae_representation>();
  REQUIRE(host.get_total_rows() == 64);
  REQUIRE(host.get_column_chunks().size() == 2);
  REQUIRE(host.get_host_data()->size() == frame_bytes * 2);

  host.mark_h2d_complete();
  batch.reset();
  output.reset();
  REQUIRE(host_space->get_total_reserved_memory() == 0);
  REQUIRE(scan.get_next_task_hint().has_value());

  REQUIRE(global->try_claim_task());
  auto eof_local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(capacity);
  sirius::op::scan::mo_native_scan_task eof_task(2, &repository, std::move(eof_local), global);
  REQUIRE_FALSE(eof_task.compute_task(cudf::get_default_stream()));
  REQUIRE(source->next_calls == 3);
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());
  REQUIRE_FALSE(global->try_claim_task());
}

TEST_CASE("GPU MO scan leaves a frame pending when the bounded source batch is full",
          "[mo_native_scan][backpressure][batching]")
{
  auto host_space = make_host_space();
  auto source     = std::make_shared<counting_native_source>();
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  const auto frame_bytes = source->batches.front()->columns().front().encoded.size();
  const auto capacity    = frame_bytes + 1;
  auto global            = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get(), capacity, capacity);
  cucascade::shared_data_repository repository;

  auto run_task = [&](std::uint64_t task_id) {
    REQUIRE(global->try_claim_task());
    auto reservation = host_space->make_reservation_or_null(capacity);
    REQUIRE(reservation);
    auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(capacity);
    local->set_reservation(std::move(reservation));
    sirius::op::scan::mo_native_scan_task task(task_id, &repository, std::move(local), global);
    auto output = task.compute_task(cudf::get_default_stream());
    REQUIRE(output);
    auto& pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output);
    auto batch         = pipelineable.get_data_batches().front();
    auto& host         = batch->get_data()->cast<sirius::host_tae_representation>();
    REQUIRE(host.get_total_rows() == 32);
    REQUIRE(host.get_host_data()->size() == frame_bytes);
    host.mark_h2d_complete();
    batch.reset();
    output.reset();
  };

  run_task(1);
  REQUIRE(source->next_calls == 2);
  REQUIRE(source->acknowledged == std::vector<std::uint64_t>{1});
  REQUIRE(scan.get_next_task_hint().has_value());

  run_task(2);
  REQUIRE(source->next_calls == 4);
  REQUIRE(source->acknowledged == std::vector<std::uint64_t>{1, 2});
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());
}

TEST_CASE("GPU MO scan bounds source batches by frame count",
          "[mo_native_scan][backpressure][batching]")
{
  auto host_space = make_host_space();
  auto source     = std::make_shared<counting_native_source>();
  source->batches.clear();
  for (std::size_t sequence = 1; sequence <= sirius::offload::max_native_frames_per_source_task + 1;
       ++sequence) {
    source->batches.push_back(
      std::make_shared<fixture_native_batch>(sequence, static_cast<std::int64_t>(sequence)));
  }
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  constexpr std::size_t capacity = 64U * 1024U;
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get(), capacity, capacity);
  REQUIRE(global->try_claim_task());

  auto reservation = host_space->make_reservation_or_null(capacity);
  REQUIRE(reservation);
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(capacity);
  local->set_reservation(std::move(reservation));
  cucascade::shared_data_repository repository;
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);

  auto output = task.compute_task(cudf::get_default_stream());
  REQUIRE(output);
  REQUIRE(source->next_calls == sirius::offload::max_native_frames_per_source_task);
  REQUIRE(source->acknowledged.size() == sirius::offload::max_native_frames_per_source_task);
  REQUIRE(source->next == sirius::offload::max_native_frames_per_source_task);
  auto& pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output);
  auto batch         = pipelineable.get_data_batches().front();
  auto& host         = batch->get_data()->cast<sirius::host_tae_representation>();
  REQUIRE(host.get_total_rows() == 32 * sirius::offload::max_native_frames_per_source_task);
  REQUIRE(host.get_column_chunks().size() == sirius::offload::max_native_frames_per_source_task);
  host.mark_h2d_complete();
  batch.reset();
  output.reset();
  REQUIRE(scan.get_next_task_hint().has_value());
}

TEST_CASE("GPU MO scan batches across non-word-aligned frame boundaries",
          "[mo_native_scan][backpressure][batching]")
{
  auto host_space = make_host_space();
  auto source     = std::make_shared<counting_native_source>();
  source->batches = {std::make_shared<fixture_native_batch>(1, 11, 33),
                     std::make_shared<fixture_native_batch>(2, 22, 33)};
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  const auto frame_bytes = source->batches.front()->columns().front().encoded.size();
  const auto capacity    = frame_bytes * 3;
  auto global            = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get(), frame_bytes * 2, capacity);
  REQUIRE(global->try_claim_task());

  auto reservation = host_space->make_reservation_or_null(capacity);
  REQUIRE(reservation);
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(capacity);
  local->set_reservation(std::move(reservation));
  cucascade::shared_data_repository repository;
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);

  auto output = task.compute_task(cudf::get_default_stream());
  REQUIRE(output);
  REQUIRE(source->next_calls == 2);
  REQUIRE(source->acknowledged == std::vector<std::uint64_t>{1, 2});
  auto& pipelineable = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output);
  auto batch         = pipelineable.get_data_batches().front();
  auto& host         = batch->get_data()->cast<sirius::host_tae_representation>();
  REQUIRE(host.get_total_rows() == 66);
  REQUIRE(host.get_column_chunks().size() == 2);
  host.mark_h2d_complete();
  batch.reset();
  output.reset();
  REQUIRE(scan.get_next_task_hint().has_value());
}

TEST_CASE("batched MO null masks preserve arbitrary frame row offsets",
          "[mo_native_scan][null_mask][batching]")
{
  auto stream = cudf::get_default_stream();

  // Frame one has 33 rows, so frame two begins one bit into a destination
  // word. MO uses 1=NULL while cuDF uses 1=VALID.
  // Each bitmap starts with count/coverage/byte-size. The second covers 35
  // rows of a 67-row vector; the missing trailing word represents valid rows.
  std::vector<std::uint64_t> source_words{
    3,
    33,
    8,
    (std::uint64_t{1} << 0) | (std::uint64_t{1} << 31) | (std::uint64_t{1} << 32),
    4,
    35,
    8,
    (std::uint64_t{1} << 0) | (std::uint64_t{1} << 1) | (std::uint64_t{1} << 31) |
      (std::uint64_t{1} << 34)};
  rmm::device_buffer device_source(1 + source_words.size() * sizeof(std::uint64_t), stream);
  auto* source = static_cast<std::uint8_t*>(device_source.data()) + 1;
  CUDF_CUDA_TRY(cudaMemcpyAsync(source,
                                source_words.data(),
                                source_words.size() * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  std::vector<sirius::cuda::tae::BatchedNullMaskDesc> descriptors{
    {source, 33, 0}, {source + 4 * sizeof(std::uint64_t), 67, 33}};
  rmm::device_buffer device_descriptors(
    descriptors.size() * sizeof(sirius::cuda::tae::BatchedNullMaskDesc), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_descriptors.data(),
                                descriptors.data(),
                                device_descriptors.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  std::vector<std::uint32_t> validity_words(4, std::numeric_limits<std::uint32_t>::max());
  rmm::device_buffer device_validity(validity_words.size() * sizeof(std::uint32_t), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_validity.data(),
                                validity_words.data(),
                                device_validity.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  sirius::cuda::tae::batched_invert_null_mask(
    static_cast<sirius::cuda::tae::BatchedNullMaskDesc*>(device_descriptors.data()),
    static_cast<std::uint32_t>(descriptors.size()),
    static_cast<std::uint32_t*>(device_validity.data()),
    stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(validity_words.data(),
                                device_validity.data(),
                                device_validity.size(),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  auto is_valid = [&](std::uint32_t row) {
    return (validity_words[row / 32] & (std::uint32_t{1} << (row % 32))) != 0;
  };
  for (std::uint32_t row = 0; row < 100; ++row) {
    const bool expected_null =
      row == 0 || row == 31 || row == 32 || row == 33 || row == 34 || row == 64 || row == 67;
    REQUIRE(is_valid(row) == !expected_null);
  }
}

TEST_CASE("GPU MO scan discards a partial source batch when input is no longer needed",
          "[mo_native_scan][backpressure][batching]")
{
  auto host_space          = make_host_space();
  auto source              = std::make_shared<counting_native_source>();
  source->not_needed_after = 1;
  auto table_scan          = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get(), 1024, 1024);
  REQUIRE(global->try_claim_task());

  cucascade::shared_data_repository repository;
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(1024);
  local->set_reservation(host_space->make_reservation_or_null(1024));
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);
  REQUIRE_FALSE(task.compute_task(cudf::get_default_stream()));
  REQUIRE(source->next_calls == 2);
  REQUIRE(source->acknowledged == std::vector<std::uint64_t>{1});
  REQUIRE(host_space->get_total_reserved_memory() == 0);
  REQUIRE_FALSE(global->try_claim_task());
}

TEST_CASE("GPU MO scan expands a constant frame within bounded staging",
          "[mo_native_scan][backpressure][batching]")
{
  auto host_space = make_host_space();

  auto source     = std::make_shared<counting_native_source>();
  source->batches = {std::make_shared<fixture_native_batch>(1, 7, 3, true)};
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  constexpr std::size_t expanded_bytes = 58;
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get(), expanded_bytes, expanded_bytes);
  REQUIRE(global->try_claim_task());

  auto reservation = host_space->make_reservation_or_null(expanded_bytes);
  REQUIRE(reservation);
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(expanded_bytes);
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
  REQUIRE(host.get_host_data()->size() == expanded_bytes);
  REQUIRE(source->next_calls == 1);
  REQUIRE(source->acknowledged == std::vector<std::uint64_t>{1});
  REQUIRE_FALSE(scan.get_next_task_hint().has_value());

  host.mark_h2d_complete();
  batch.reset();
  output.reset();
  REQUIRE(scan.get_next_task_hint().has_value());
}

TEST_CASE("GPU MO scan clears stale NULL string descriptors only in its owned buffer",
          "[mo_native_scan][backpressure][null_mask]")
{
  auto host_space = make_host_space(32);
  auto source     = std::make_shared<counting_native_source>();
  auto input      = fixture_native_batch::null_string();
  const std::string original(input->columns()[0].encoded);
  source->batches = {input};
  auto table_scan = make_stream_table_scan(source);
  sirius::op::sirius_physical_gpu_mo_scan scan(table_scan.get());
  auto global = std::make_shared<sirius::op::scan::mo_native_scan_task_global_state>(
    nullptr, &scan, host_space.get(), 128, 128);
  REQUIRE(global->try_claim_task());
  auto local = std::make_unique<sirius::op::scan::mo_native_scan_task_local_state>(128);
  local->set_reservation(host_space->make_reservation_or_null(128));
  cucascade::shared_data_repository repository;
  sirius::op::scan::mo_native_scan_task task(1, &repository, std::move(local), global);
  auto stream = cudf::get_default_stream();
  auto output = task.compute_task(stream);
  REQUIRE(output);
  auto& data = dynamic_cast<sirius::op::pipelineable_operator_data&>(*output);
  auto batch = data.get_data_batches().front();
  auto& host = batch->get_data()->cast<sirius::host_tae_representation>();
  rmm::device_buffer device(host.get_host_data()->size(), stream);
  CUDF_CUDA_TRY(host.get_host_data()->copy_to_device(device.data(), stream));
  std::string copied(device.size(), 0);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    copied.data(), device.data(), copied.size(), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  auto expected = original;
  expected.replace(1 + sizeof(tae::MOType) + 8, tae::VARLENA_SIZE, tae::VARLENA_SIZE, '\0');
  REQUIRE(copied == expected);
  REQUIRE(input->columns()[0].encoded == original);
  host.mark_h2d_complete();
  batch.reset();
  output.reset();
  REQUIRE(host_space->get_total_reserved_memory() == 0);
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
