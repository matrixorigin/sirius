/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "sirius_c.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace sirius::embedding {

// Query-owned counters outlive the runtime owners that feed them. All values
// are monotonic except live retained/active/parked gauges; the query coordinator
// keeps this object until the C query handle is successfully closed.
class execution_stats final {
 public:
  void add_source(uint32_t source_kind) noexcept;
  void terminal(sirius_status status, bool fatal) noexcept;

  void gpu_task_started() noexcept;
  void gpu_task_completed() noexcept;

  void mo_input_retain(std::size_t charged_bytes) noexcept;
  void mo_input_release(std::size_t charged_bytes) noexcept;
  void mo_input_blocked() noexcept;
  void mo_input_unit() noexcept;

  void tae_configure(std::size_t limit, std::size_t slice_bytes) noexcept;
  void tae_request(std::size_t queued) noexcept;
  void tae_issue() noexcept;
  void tae_complete() noexcept;
  void tae_cache(std::size_t charged_bytes) noexcept;
  void tae_staging(std::size_t charged_bytes) noexcept;
  void tae_gpu_admission_wait() noexcept;
  void tae_gpu_reservation(std::size_t admitted_bytes) noexcept;
  void tae_payload(std::size_t bytes) noexcept;

  void result_retain(std::size_t charged_bytes) noexcept;
  void result_release(std::size_t charged_bytes) noexcept;
  void result_publish(uint64_t rows, std::size_t payload_bytes) noexcept;
  void result_blocked() noexcept;
  void result_parked(bool value) noexcept;

  [[nodiscard]] sirius_query_execution_stats inspect() const noexcept;

 private:
  static void peak(std::atomic<uint64_t>& target, uint64_t value) noexcept;
  [[nodiscard]] bool has_source(uint32_t source) const noexcept;

  std::atomic<uint32_t> source_mask_{0};
  std::atomic<uint32_t> terminal_{0};
  std::atomic<sirius_status> terminal_status_{SIRIUS_OK};
  std::atomic<uint32_t> fatal_{0};

  std::atomic<uint64_t> gpu_tasks_started_{0};
  std::atomic<uint64_t> gpu_tasks_completed_{0};

  std::atomic<uint64_t> mo_input_units_{0};
  std::atomic<uint64_t> mo_input_retained_{0};
  std::atomic<uint64_t> mo_input_peak_{0};
  std::atomic<uint64_t> mo_input_blocked_{0};

  std::atomic<uint64_t> tae_requests_{0};
  std::atomic<uint64_t> tae_issued_{0};
  std::atomic<uint64_t> tae_completed_{0};
  std::atomic<uint64_t> tae_active_{0};
  std::atomic<uint64_t> tae_peak_active_{0};
  std::atomic<uint64_t> tae_peak_queued_{0};
  std::atomic<uint64_t> tae_limit_{0};
  std::atomic<uint64_t> tae_slice_{0};
  std::atomic<uint64_t> tae_peak_cache_{0};
  std::atomic<uint64_t> tae_peak_staging_{0};
  std::atomic<uint64_t> tae_gpu_admission_waits_{0};
  std::atomic<uint64_t> tae_peak_gpu_reservation_{0};
  std::atomic<uint64_t> tae_payload_bytes_{0};

  std::atomic<uint64_t> result_rows_{0};
  std::atomic<uint64_t> result_payload_bytes_{0};
  std::atomic<uint64_t> result_retained_{0};
  std::atomic<uint64_t> result_peak_{0};
  std::atomic<uint64_t> result_blocked_{0};
  std::atomic<uint64_t> result_parked_{0};
};

}  // namespace sirius::embedding
