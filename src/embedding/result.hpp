/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "embedding/input.hpp"

#include <array>

namespace sirius::embedding {
inline constexpr std::size_t result_window = 64u << 20;
inline constexpr std::size_t result_target = 32u << 20;
class native_result;
struct result_batch {
  ~result_batch();
  std::shared_ptr<native_result> owner;
  buffer_budget::lease credit;
  std::shared_ptr<input_pool> pool;
  std::unique_ptr<input_storage> storage;
  std::unique_ptr<sirius_input_vector[]> columns;
  uint32_t rows{}, column_count{};
  std::size_t payload_bytes{};
  enum class phase { filling, queued, borrowed } state{phase::filling};
};

// The ring itself is bounded independently of payload sizes. A batch's credit
// includes its object, descriptors and allocator-rounded payload storage.
class native_result : public std::enable_shared_from_this<native_result> {
 public:
  void activate(std::shared_ptr<input_pool> pool);
  std::size_t allocation_charge(std::size_t bytes, uint32_t columns) const;
  sirius_status try_allocate(std::size_t bytes,
                             uint32_t columns,
                             std::shared_ptr<result_batch>& out);
  void publish(std::shared_ptr<result_batch> batch);
  std::shared_ptr<result_batch> next(clock::time_point deadline);
  void complete(sirius_error outcome);
  void cancel();
  void subscribe(std::shared_ptr<capacity_waker> wake);
  void parked(bool value);
  sirius_result_stats inspect() const;
  bool borrowed() const;
  void released(result_batch::phase state) noexcept;

 private:
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  buffer_budget budget_{result_window, 128};
  std::shared_ptr<input_pool> pool_;
  std::array<std::shared_ptr<result_batch>, 128> queue_{};
  std::size_t head_{}, size_{}, filling_{}, borrowed_{}, parked_{}, blocked_{};
  bool ended_{false}, pulling_{false};
  sirius_error outcome_{};
};
}  // namespace sirius::embedding
