/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <data/host_tae_representation.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <embedding/buffer_budget.hpp>
#include <embedding/tae_demand.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/tae_scan_plan.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace sirius::op {
class sirius_dynamic_filter_set;
}

namespace sirius::op::scan {

class tae_ingestible_table_info final : public ingestible_table_info {
 public:
  duckdb::unique_ptr<tae::TAEScanBindData> bind_data;
  bool embedded_manifest{false};
  std::shared_ptr<sirius::embedding::buffer_budget> embedded_host_budget;
  std::shared_ptr<sirius::embedding::tae_demand_controller> embedded_controller;
  io::ioctx_resolver embedded_resolve;
  duckdb::vector<sirius::logical_type> returned_types;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<sirius::logical_type> output_types;
  duckdb::vector<std::string> names;
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> sirius_dynamic_filters;
  duckdb::ClientContext* context = nullptr;

  [[nodiscard]] std::span<std::string const> column_names() const override { return names; }
  // Diagnostics identify the scan kind without exposing manifest/object paths.
  [[nodiscard]] std::string display_name() const override { return "tae_scan"; }
  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    return bind_data ? std::span<std::string const>(object_paths_) : std::span<std::string const>{};
  }

  void refresh_object_paths();

 private:
  std::vector<std::string> object_paths_;
};

/// A scanner-produced compressed object ready for GPU LZ4 decode.
class tae_scan_info final : public scan_info {
 public:
  // Declared before the host owner so destruction frees payload bytes first,
  // then returns staging credit.
  std::shared_ptr<sirius::embedding::buffer_budget::lease> host_credit;
  std::shared_ptr<sirius::pinned_host_buffer> host_data;
  std::shared_ptr<sirius::embedding::tae_work_permit> work_permit;
  mutable std::shared_ptr<sirius::pinned_host_buffer> staging;
  std::shared_ptr<io::ioctx> io_context;
  std::string object_path;
  std::uint64_t object_size{0};
  bool crc_wrapped{false};
  std::stop_token stop;
  std::size_t gpu_peak_bytes{0};
  mutable std::atomic<std::size_t> admitted_bytes{0};
  std::vector<sirius::host_tae_representation::column_chunk_info> chunks;
  std::size_t rows               = 0;
  std::size_t compressed_bytes   = 0;
  std::size_t uncompressed_bytes = 0;

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override { return uncompressed_bytes; }
  [[nodiscard]] std::size_t estimated_working_set_bytes() const noexcept override
  {
    return gpu_peak_bytes ? gpu_peak_bytes : uncompressed_bytes;
  }
  [[nodiscard]] std::size_t mandatory_gpu_reservation_bytes() const noexcept override
  {
    return gpu_peak_bytes;
  }
  void gpu_admitted(std::size_t bytes) const override;
  std::shared_ptr<void> execution_lease() const override { return work_permit; }
};

/// Native TAE source for the current GPU_SCAN + scan-manager architecture.
class tae_gpu_ingestible final : public gpu_ingestible {
 public:
  explicit tae_gpu_ingestible(std::unique_ptr<tae_ingestible_table_info> info);
  ~tae_gpu_ingestible() override;
  void stop_metadata_scan() noexcept override;
  bool is_live() const noexcept override { return _info->embedded_manifest; }
  bool live_ready() const override;
  bool live_exhausted() const override;
  void live_request() override;
  void live_set_io_resolver(io::ioctx_resolver) override;
  void live_subscribe(std::shared_ptr<embedding::capacity_waker>) override;
  void live_stop() override;
  std::unique_ptr<op::operator_data> live_claim() override;

  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;
  [[nodiscard]] bool has_processed_all_metadata() const override;
  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override;

  filtered_table materialize_metadata_to_table(
    scan_info const& info,
    const cucascade::memory::memory_space& mem_space,
    ::cuda::stream_ref stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    const cucascade::memory::memory_space& mem_space,
    ::cuda::stream_ref stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
    std::unique_ptr<cudf::column>* survivors,
    std::span<std::size_t const> elided) override;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept override { return *_info; }
  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override;
  [[nodiscard]] bool has_row_filter() const noexcept override
  {
    return _filter_expression != nullptr;
  }
  [[nodiscard]] bool can_report_survivors() const noexcept override { return true; }

 private:
  void produce_live(std::shared_ptr<embedding::tae_work_permit>) noexcept;
  std::unique_ptr<tae_scan_info> plan_live_split(
    std::shared_ptr<embedding::tae_work_permit> const&);
  [[nodiscard]] std::unique_ptr<tae_scan_info> load_object(
    std::size_t object_index,
    std::size_t block_index,
    std::shared_ptr<io::ioctx> const& io_ctx) const;
  [[nodiscard]] std::unique_ptr<cudf::table> make_empty_table(
    const cucascade::memory::memory_space& mem_space, rmm::cuda_stream_view stream) const;

  std::unique_ptr<tae_ingestible_table_info> _info;
  tae_scan_plan _plan;
  std::shared_ptr<duckdb::Expression> _filter_expression;
  mutable std::mutex _work_mutex;
  std::size_t _next_object{0};
  std::size_t _next_block{0};
  std::stop_source _metadata_stop;
  std::shared_ptr<embedding::capacity_waker> _waker;
  std::unique_ptr<tae_scan_info> _ready;
  std::exception_ptr _live_error;
  bool _pending{false};
  bool _exhausted{false};
  bool _demand_started{false};
};

std::shared_ptr<tae_gpu_ingestible> make_ingestible(
  std::unique_ptr<tae_ingestible_table_info> info);

}  // namespace sirius::op::scan
