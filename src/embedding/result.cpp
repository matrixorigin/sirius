/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/result.hpp"

#include <algorithm>

namespace sirius::embedding {
result_batch::~result_batch()
{
  // Retire physical ownership, then the public gauge, before returned credit
  // wakes a publisher that can charge the same window again.
  storage.reset();
  columns.reset();
  pool.reset();
  if (stats) stats->result_release(charged_bytes);
  credit.reset();
  if (owner) owner->released(state);
}
void native_result::activate(std::shared_ptr<input_pool> pool)
{
  std::lock_guard lock(mutex_);
  if (!pool || pool_ || ended_) throw failure(SIRIUS_INVALID_STATE, "invalid result activation");
  pool_ = std::move(pool);
}
std::size_t native_result::allocation_charge(std::size_t bytes, uint32_t columns) const
{
  std::lock_guard lock(mutex_);
  if (ended_)
    throw failure(outcome_.code ? outcome_.code : SIRIUS_CANCELLED,
                  "native result publication has ended");
  if (!pool_) throw failure(SIRIUS_INVALID_STATE, "native result is not activated");
  if (bytes > result_window || columns > input_columns_limit) return result_window + 1;
  auto rounded = bytes ? pool_->rounded(bytes) : 0;
  return rounded + sizeof(result_batch) + columns * sizeof(sirius_input_vector);
}
sirius_status native_result::try_allocate(std::size_t bytes,
                                          uint32_t columns,
                                          std::shared_ptr<result_batch>& out)
{
  if (out || columns > input_columns_limit) return SIRIUS_INVALID_ARGUMENT;
  if (bytes > result_window) return SIRIUS_RESOURCE_EXHAUSTED;
  std::shared_ptr<input_pool> pool;
  std::size_t charge{};
  {
    std::lock_guard lock(mutex_);
    if (ended_) return outcome_.code ? outcome_.code : SIRIUS_CANCELLED;
    if (!pool_) return SIRIUS_INVALID_STATE;
    pool = pool_;
    // Capture accounting with the pool: cancellation can detach the facade's
    // reservation immediately after this lock is released.
    charge = (bytes ? pool->rounded(bytes) : 0) + sizeof(result_batch) +
             columns * sizeof(sirius_input_vector);
  }
  if (charge > result_window) return SIRIUS_RESOURCE_EXHAUSTED;
  buffer_budget::lease credit;
  auto status = budget_.try_acquire(charge, credit);
  if (status != SIRIUS_OK) {
    if (status == SIRIUS_TIMEOUT) {
      std::lock_guard lock(mutex_);
      ++blocked_;
      if (stats_) stats_->result_blocked();
    }
    return status;
  }
  auto batch           = std::make_shared<result_batch>();
  batch->stats         = stats_;
  batch->charged_bytes = charge;
  if (batch->stats) batch->stats->result_retain(charge);
  batch->credit = std::move(credit);
  batch->pool   = pool;
  if (bytes) batch->storage = pool->allocate(bytes);
  batch->columns       = std::make_unique<sirius_input_vector[]>(columns);
  batch->column_count  = columns;
  batch->payload_bytes = bytes;
  {
    std::lock_guard lock(mutex_);
    if (ended_) return outcome_.code ? outcome_.code : SIRIUS_CANCELLED;
    ++filling_;
    batch->owner = shared_from_this();
  }
  out = std::move(batch);
  return SIRIUS_OK;
}
void native_result::publish(std::shared_ptr<result_batch> batch)
{
  {
    std::lock_guard lock(mutex_);
    if (ended_) throw failure(outcome_.code ? outcome_.code : SIRIUS_CANCELLED, "result closed");
    if (!batch || batch->owner.get() != this || batch->state != result_batch::phase::filling ||
        size_ == queue_.size())
      throw failure(SIRIUS_INVALID_STATE, "invalid result publication");
    auto const rows    = batch->rows;
    auto const payload = batch->payload_bytes;
    batch->state       = result_batch::phase::queued;
    --filling_;
    queue_[(head_ + size_) % queue_.size()] = std::move(batch);
    ++size_;
    if (stats_) stats_->result_publish(rows, payload);
  }
  changed_.notify_all();
}
std::shared_ptr<result_batch> native_result::next(clock::time_point deadline)
{
  std::unique_lock lock(mutex_);
  if (pulling_) throw failure(SIRIUS_BUSY, "a result pull is already pending");
  pulling_ = true;
  struct reset_flag {
    bool& value;
    ~reset_flag() { value = false; }
  } reset{pulling_};
  if (!changed_.wait_until(lock, deadline, [&] { return size_ || ended_; }))
    throw failure(SIRIUS_TIMEOUT, "waiting for native result timed out");
  if (outcome_.code) throw failure(outcome_.code, outcome_.message);
  if (!size_) throw failure(SIRIUS_EOF, "native result ended");
  auto batch = std::move(queue_[head_]);
  head_      = (head_ + 1) % queue_.size();
  --size_;
  ++borrowed_;
  batch->state = result_batch::phase::borrowed;
  return batch;
}
void native_result::complete(sirius_error outcome)
{
  std::array<std::shared_ptr<result_batch>, 128> discarded;
  {
    std::lock_guard lock(mutex_);
    // A fatal cleanup outcome must not be hidden by an earlier cancellation.
    if (!ended_ || outcome.code == SIRIUS_GPU_UNAVAILABLE ||
        (outcome_.code == SIRIUS_CANCELLED && outcome.code && outcome.code != SIRIUS_CANCELLED))
      outcome_ = outcome;
    ended_ = true;
    // Do not carry an uncounted reservation past the terminal-state lock.
    // Filling/queued/borrowed batches retain their own pools and leases.
    pool_.reset();
    if (outcome_.code || outcome.code) {
      discarded.swap(queue_);
      size_ = 0;
    }
  }
  budget_.close();
  changed_.notify_all();
}
void native_result::cancel()
{
  sirius_error error{};
  assign_error(error, SIRIUS_CANCELLED, "native result cancelled");
  complete(error);
}
void native_result::subscribe(std::shared_ptr<capacity_waker> wake)
{
  budget_.set_waker(std::move(wake));
}
void native_result::parked(bool value)
{
  std::lock_guard lock(mutex_);
  if (value)
    ++parked_;
  else {
    assert(parked_);
    --parked_;
  }
  if (stats_) stats_->result_parked(value);
}
void native_result::released(result_batch::phase state) noexcept
{
  std::lock_guard lock(mutex_);
  if (state == result_batch::phase::filling) {
    assert(filling_);
    --filling_;
  }
  if (state == result_batch::phase::borrowed) {
    assert(borrowed_);
    --borrowed_;
  }
}
bool native_result::borrowed() const
{
  std::lock_guard lock(mutex_);
  return borrowed_ != 0;
}
sirius_result_stats native_result::inspect() const
{
  std::lock_guard lock(mutex_);
  auto state = budget_.inspect();
  return {sizeof(sirius_result_stats),
          SIRIUS_ABI_VERSION,
          state.bytes,
          state.peak,
          state.leases,
          size_,
          borrowed_,
          filling_,
          parked_,
          blocked_};
}
}  // namespace sirius::embedding
