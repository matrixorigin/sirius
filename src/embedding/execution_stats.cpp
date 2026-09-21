/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/execution_stats.hpp"

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace sirius::embedding {

static_assert(std::is_standard_layout_v<sirius_query_execution_stats>);
static_assert(sizeof(sirius_query_execution_stats) == 224);
static_assert(offsetof(sirius_query_execution_stats, struct_size) == 0);
static_assert(offsetof(sirius_query_execution_stats, abi_version) == 4);
static_assert(offsetof(sirius_query_execution_stats, source_mask) == 8);
static_assert(offsetof(sirius_query_execution_stats, terminal) == 12);
static_assert(offsetof(sirius_query_execution_stats, terminal_status) == 16);
static_assert(offsetof(sirius_query_execution_stats, fatal) == 20);
static_assert(offsetof(sirius_query_execution_stats, gpu_tasks_started) == 24);
static_assert(offsetof(sirius_query_execution_stats, result_parked_publications) == 216);

void execution_stats::peak(std::atomic<uint64_t>& target, uint64_t value) noexcept
{
  auto current = target.load(std::memory_order_relaxed);
  while (current < value &&
         !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {}
}

void execution_stats::add_source(uint32_t source_kind) noexcept
{
  uint32_t bit = 0;
  if (source_kind == SIRIUS_READ_MO) bit = SIRIUS_QUERY_SOURCE_MO;
  if (source_kind == SIRIUS_READ_TAE) bit = SIRIUS_QUERY_SOURCE_TAE;
  source_mask_.fetch_or(bit, std::memory_order_relaxed);
}
bool execution_stats::has_source(uint32_t source) const noexcept
{
  return (source_mask_.load(std::memory_order_relaxed) & source) != 0;
}

void execution_stats::terminal(sirius_status status, bool fatal) noexcept
{
  uint32_t expected = 0;
  if (!terminal_.compare_exchange_strong(
        expected, 2, std::memory_order_acq_rel, std::memory_order_acquire))
    return;
  terminal_status_.store(status, std::memory_order_relaxed);
  fatal_.store(fatal, std::memory_order_relaxed);
  terminal_.store(1, std::memory_order_release);
}

void execution_stats::gpu_task_started() noexcept
{
  gpu_tasks_started_.fetch_add(1, std::memory_order_relaxed);
}
void execution_stats::gpu_task_completed() noexcept
{
  gpu_tasks_completed_.fetch_add(1, std::memory_order_relaxed);
}

void execution_stats::mo_input_retain(std::size_t charged_bytes) noexcept
{
  auto retained =
    mo_input_retained_.fetch_add(charged_bytes, std::memory_order_relaxed) + charged_bytes;
  peak(mo_input_peak_, retained);
}
void execution_stats::mo_input_release(std::size_t charged_bytes) noexcept
{
  auto previous = mo_input_retained_.fetch_sub(charged_bytes, std::memory_order_relaxed);
  assert(previous >= charged_bytes);
}
void execution_stats::mo_input_blocked() noexcept
{
  mo_input_blocked_.fetch_add(1, std::memory_order_relaxed);
}
void execution_stats::mo_input_unit() noexcept
{
  mo_input_units_.fetch_add(1, std::memory_order_relaxed);
}

void execution_stats::tae_configure(std::size_t limit, std::size_t slice_bytes) noexcept
{
  tae_limit_.store(limit, std::memory_order_relaxed);
  tae_slice_.store(slice_bytes, std::memory_order_relaxed);
}
void execution_stats::tae_request(std::size_t queued) noexcept
{
  tae_requests_.fetch_add(1, std::memory_order_relaxed);
  peak(tae_peak_queued_, queued);
}
void execution_stats::tae_issue() noexcept
{
  tae_issued_.fetch_add(1, std::memory_order_relaxed);
  auto const active = tae_active_.fetch_add(1, std::memory_order_relaxed) + 1;
  peak(tae_peak_active_, active);
}
void execution_stats::tae_complete() noexcept
{
  tae_completed_.fetch_add(1, std::memory_order_relaxed);
  auto const previous = tae_active_.fetch_sub(1, std::memory_order_relaxed);
  assert(previous != 0);
}
void execution_stats::tae_cache(std::size_t charged_bytes) noexcept
{
  peak(tae_peak_cache_, charged_bytes);
}
void execution_stats::tae_staging(std::size_t charged_bytes) noexcept
{
  peak(tae_peak_staging_, charged_bytes);
}
void execution_stats::tae_gpu_admission_wait() noexcept
{
  if (!has_source(SIRIUS_QUERY_SOURCE_TAE)) return;
  tae_gpu_admission_waits_.fetch_add(1, std::memory_order_relaxed);
}
void execution_stats::tae_gpu_reservation(std::size_t admitted_bytes) noexcept
{
  if (!has_source(SIRIUS_QUERY_SOURCE_TAE)) return;
  peak(tae_peak_gpu_reservation_, admitted_bytes);
}
void execution_stats::tae_payload(std::size_t bytes) noexcept
{
  tae_payload_bytes_.fetch_add(bytes, std::memory_order_relaxed);
}

void execution_stats::result_retain(std::size_t charged_bytes) noexcept
{
  auto retained =
    result_retained_.fetch_add(charged_bytes, std::memory_order_relaxed) + charged_bytes;
  peak(result_peak_, retained);
}
void execution_stats::result_release(std::size_t charged_bytes) noexcept
{
  auto previous = result_retained_.fetch_sub(charged_bytes, std::memory_order_relaxed);
  assert(previous >= charged_bytes);
}
void execution_stats::result_publish(uint64_t rows, std::size_t payload_bytes) noexcept
{
  result_rows_.fetch_add(rows, std::memory_order_relaxed);
  result_payload_bytes_.fetch_add(payload_bytes, std::memory_order_relaxed);
}
void execution_stats::result_blocked() noexcept
{
  result_blocked_.fetch_add(1, std::memory_order_relaxed);
}
void execution_stats::result_parked(bool value) noexcept
{
  if (value) {
    result_parked_.fetch_add(1, std::memory_order_relaxed);
  } else {
    auto previous = result_parked_.fetch_sub(1, std::memory_order_relaxed);
    assert(previous != 0);
  }
}

sirius_query_execution_stats execution_stats::inspect() const noexcept
{
  auto const state    = terminal_.load(std::memory_order_acquire);
  auto const terminal = state == 1;
  auto const status   = terminal ? terminal_status_.load(std::memory_order_relaxed) : SIRIUS_OK;
  auto const fatal    = terminal ? fatal_.load(std::memory_order_relaxed) : 0;
  return {sizeof(sirius_query_execution_stats),
          SIRIUS_ABI_VERSION,
          source_mask_.load(std::memory_order_relaxed),
          terminal,
          status,
          fatal,
          gpu_tasks_started_.load(std::memory_order_relaxed),
          gpu_tasks_completed_.load(std::memory_order_relaxed),
          mo_input_units_.load(std::memory_order_relaxed),
          mo_input_retained_.load(std::memory_order_relaxed),
          mo_input_peak_.load(std::memory_order_relaxed),
          mo_input_blocked_.load(std::memory_order_relaxed),
          tae_requests_.load(std::memory_order_relaxed),
          tae_issued_.load(std::memory_order_relaxed),
          tae_completed_.load(std::memory_order_relaxed),
          tae_active_.load(std::memory_order_relaxed),
          tae_peak_active_.load(std::memory_order_relaxed),
          tae_peak_queued_.load(std::memory_order_relaxed),
          tae_limit_.load(std::memory_order_relaxed),
          tae_slice_.load(std::memory_order_relaxed),
          tae_peak_cache_.load(std::memory_order_relaxed),
          tae_peak_staging_.load(std::memory_order_relaxed),
          tae_gpu_admission_waits_.load(std::memory_order_relaxed),
          tae_peak_gpu_reservation_.load(std::memory_order_relaxed),
          tae_payload_bytes_.load(std::memory_order_relaxed),
          result_rows_.load(std::memory_order_relaxed),
          result_payload_bytes_.load(std::memory_order_relaxed),
          result_retained_.load(std::memory_order_relaxed),
          result_peak_.load(std::memory_order_relaxed),
          result_blocked_.load(std::memory_order_relaxed),
          result_parked_.load(std::memory_order_relaxed)};
}

}  // namespace sirius::embedding
