/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "embedding/buffer_budget.hpp"
#include "embedding/control.hpp"

#include <atomic>
#include <functional>
#include <span>
#include <vector>

namespace sirius::embedding {
inline constexpr std::size_t input_window        = 64u << 20;
inline constexpr std::size_t input_target        = 32u << 20;
inline constexpr std::size_t input_columns_limit = 1024;

// The GPU implementation uses reserved cuCascade pinned blocks. CPU tests use
// the same lifecycle with a test allocator, never a second public SQL path.
struct input_storage {
  virtual ~input_storage()                                                          = default;
  virtual std::size_t size() const                                                  = 0;
  virtual void visit(std::function<void(std::size_t, std::span<std::byte>)> const&) = 0;
  void write(std::size_t offset, std::span<const std::byte> bytes);
  virtual void read(std::size_t offset, std::span<std::byte> bytes) const;
};
struct input_pool {
  virtual ~input_pool()                                              = default;
  virtual std::size_t rounded(std::size_t bytes) const               = 0;
  virtual std::unique_ptr<input_storage> allocate(std::size_t bytes) = 0;
};
class native_input;
struct input_batch {
  ~input_batch();
  std::shared_ptr<native_input> owner;
  buffer_budget::lease credit;
  std::shared_ptr<input_pool> pool;
  std::unique_ptr<input_storage> storage;
  std::vector<sirius_input_vector> columns;
  uint32_t rows{0};
  std::size_t payload_bytes{0};
  bool published{false};
  void write(std::size_t offset, std::span<const std::byte> bytes);
  bool is_null(std::size_t column, uint32_t row) const;
  std::pair<std::size_t, std::size_t> string_range(std::size_t column, uint32_t row) const;
};
struct input_slice {
  std::shared_ptr<input_batch> batch;
  uint32_t begin, rows;
};
struct input_unit {
  // Destruction returns expanded-source credit only after all host owners.
  buffer_budget::lease expanded_credit;
  std::vector<input_slice> slices;
  std::vector<std::size_t> chars;
  std::vector<std::size_t> nulls;
  std::size_t rows{0}, bytes{0};
};

class native_input : public std::enable_shared_from_this<native_input> {
 public:
  native_input(uint64_t id,
               std::vector<sirius_input_column> schema,
               std::stop_token stop,
               clock::time_point deadline,
               std::size_t capacity = input_window,
               std::size_t count    = 128);
  void activate(std::shared_ptr<input_pool> pool);
  std::shared_ptr<input_batch> acquire(std::size_t bytes, clock::time_point wait_until);
  void publish(std::shared_ptr<input_batch> const&,
               uint32_t rows,
               std::span<const sirius_input_vector> columns);
  void finish();
  void stop(sirius_status code = SIRIUS_CANCELLED, const char* message = "native input cancelled");
  void discard();  // coordinator only, after stop; never frees filling/claimed leases
  void subscribe(std::shared_ptr<capacity_waker> wake);
  std::unique_ptr<input_unit> claim();  // nonblocking; nullptr means no ready unit
  bool ready() const;
  bool exhausted() const;
  bool finished() const;
  sirius_error outcome() const;
  sirius_input_stats inspect() const;
  void release_filling();
  std::vector<sirius_input_column> const schema;
  uint64_t const id;
  std::size_t const capacity;

 private:
  void check_open() const;
  void notify() const;
  std::stop_token stop_;
  clock::time_point deadline_;
  mutable std::mutex mutex_;
  std::mutex claim_mutex_;
  std::shared_ptr<input_pool> pool_;
  buffer_budget budget_;
  buffer_budget expanded_{input_window, 128};
  std::deque<std::shared_ptr<input_batch>> queue_;
  uint32_t front_row_{0};
  std::size_t filling_{0};
  std::atomic<std::size_t> blocked_{0};
  bool eos_{false};
  bool had_rows_{false}, empty_claimed_{false};
  sirius_error error_{};
  std::shared_ptr<capacity_waker> waker_;
  std::stop_callback<std::function<void()>> cancel_;
};

// Schema frozen by engine_control when preparation is queued. Input state has
// its own lock; no producer allocation/copy occurs under the coordinator lock.
struct input_registry {
  std::vector<std::shared_ptr<native_input>> reads;
  std::atomic<std::size_t> handles{0}, filling_handles{0};
  void stop();
  void discard();
  sirius_error outcome() const;
};
std::size_t input_element_size(uint32_t oid);
bool input_string_type(uint32_t oid);
void validate_input_schema(std::span<const sirius_input_column> schema);
}  // namespace sirius::embedding
