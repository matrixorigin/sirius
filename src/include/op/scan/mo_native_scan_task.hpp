/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "data/host_tae_representation.hpp"
#include "offload/mo_native_limits.hpp"
#include "op/sirius_physical_gpu_mo_scan.hpp"
#include "pipeline/sirius_pipeline_itask.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"

#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <memory>

namespace sirius::op::scan {

class mo_native_scan_task_global_state final : public pipeline::sirius_pipeline_task_global_state {
 public:
  mo_native_scan_task_global_state(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
                                   sirius_physical_gpu_mo_scan* scan_op,
                                   cucascade::memory::memory_space* host_memory_space);

  bool try_claim_task() noexcept;
  void finish_eof() noexcept;
  void acknowledge(std::uint64_t sequence) noexcept;
  void release_after_h2d() noexcept;

  [[nodiscard]] sirius_physical_gpu_mo_scan& get_operator() const { return *_scan_op; }
  [[nodiscard]] cucascade::memory::memory_space* get_host_memory_space() const
  {
    return _host_memory_space;
  }

 private:
  sirius_physical_gpu_mo_scan* _scan_op;
  cucascade::memory::memory_space* _host_memory_space;
};

class mo_native_scan_task_local_state final : public pipeline::sirius_pipeline_task_local_state {
 public:
  [[nodiscard]] std::size_t get_task_consumption_basis() const override
  {
    return offload::mo_native_scan_reservation_bytes();
  }
};

class mo_native_scan_task final : public pipeline::sirius_pipeline_itask {
 public:
  mo_native_scan_task(std::uint64_t task_id,
                      cucascade::shared_data_repository* data_repo,
                      std::unique_ptr<mo_native_scan_task_local_state> local_state,
                      std::shared_ptr<mo_native_scan_task_global_state> global_state);
  ~mo_native_scan_task() override;

  void execute(rmm::cuda_stream_view stream) override;
  std::unique_ptr<operator_data> compute_task(rmm::cuda_stream_view stream) override;
  void publish_output(operator_data& output_data, rmm::cuda_stream_view stream) override;
  [[nodiscard]] std::size_t get_estimated_reservation_size() const override;
  std::vector<sirius_physical_operator*> get_output_consumers() override;

 private:
  std::uint64_t _task_id;
  cucascade::shared_data_repository* _data_repo;
};

}  // namespace sirius::op::scan
