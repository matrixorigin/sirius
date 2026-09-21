/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sirius_c.h"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <utility>

namespace sirius::embedding {

// Scheduler notifications must be nonblocking and must not borrow raw operators.
// Implementations hold a weak query owner so old leases cannot wake a new query.
struct capacity_waker {
  virtual ~capacity_waker()    = default;
  virtual void wake() noexcept = 0;
};

// Strict, shared-lifetime accounting: dequeue does not release credit. A lease
// follows filling, publication, asynchronous consumption and final release.
class buffer_budget {
  struct state {
    state(std::size_t bytes, std::size_t count) : capacity(bytes), max_leases(count) {}
    const std::size_t capacity;
    const std::size_t max_leases;
    std::size_t used{0}, leases{0}, peak{0};
    bool closed{false};
    std::mutex mutex;
    std::condition_variable_any available;
    std::shared_ptr<capacity_waker> waker;
  };

 public:
  using clock = std::chrono::steady_clock;
  struct snapshot {
    std::size_t bytes, leases, peak;
    bool closed;
  };

  class lease {
   public:
    lease()                        = default;
    lease(lease const&)            = delete;
    lease& operator=(lease const&) = delete;
    lease(lease&& other) noexcept
      : owner_(std::move(other.owner_)), bytes_(std::exchange(other.bytes_, 0))
    {
    }
    lease& operator=(lease&& other) noexcept
    {
      if (this != &other) {
        reset();
        owner_ = std::move(other.owner_);
        bytes_ = std::exchange(other.bytes_, 0);
      }
      return *this;
    }
    ~lease() { reset(); }
    explicit operator bool() const noexcept { return owner_ != nullptr; }
    std::size_t bytes() const noexcept { return bytes_; }
    void reset() noexcept
    {
      auto owner = std::move(owner_);
      if (!owner) return;
      std::shared_ptr<capacity_waker> waker;
      {
        std::lock_guard lock(owner->mutex);
        assert(owner->used >= bytes_ && owner->leases != 0);
        owner->used -= bytes_;
        --owner->leases;
        if (!owner->closed) waker = owner->waker;
      }
      bytes_ = 0;
      owner->available.notify_all();
      if (waker) waker->wake();
    }

   private:
    friend class buffer_budget;
    lease(std::shared_ptr<state> owner, std::size_t bytes) noexcept
      : owner_(std::move(owner)), bytes_(bytes)
    {
    }
    std::shared_ptr<state> owner_;
    std::size_t bytes_{0};
  };

  buffer_budget(std::size_t bytes, std::size_t count)
    : state_(std::make_shared<state>(bytes, count))
  {
    if (bytes == 0 || count == 0) throw std::invalid_argument("empty buffer budget");
  }
  buffer_budget(buffer_budget const&)            = delete;
  buffer_budget& operator=(buffer_budget const&) = delete;
  ~buffer_budget() { close(); }

  // A failed acquire leaves out unchanged; no oversize/deadlock-escape grant.
  // Nonblocking native-worker admission. TIMEOUT means temporary capacity
  // pressure; RESOURCE_EXHAUSTED can never fit. Callers subscribe to the query
  // waker before checking capacity so a concurrent release cannot be lost.
  sirius_status try_acquire(std::size_t bytes, lease& out)
  {
    if (out || bytes == 0) return SIRIUS_INVALID_ARGUMENT;
    auto owner = state_;
    if (bytes > owner->capacity) return SIRIUS_RESOURCE_EXHAUSTED;
    std::lock_guard lock(owner->mutex);
    if (owner->closed) return SIRIUS_CANCELLED;
    if (owner->leases == owner->max_leases || bytes > owner->capacity - owner->used)
      return SIRIUS_TIMEOUT;
    owner->used += bytes;
    ++owner->leases;
    if (owner->used > owner->peak) owner->peak = owner->used;
    out = lease(owner, bytes);
    return SIRIUS_OK;
  }

  sirius_status acquire(std::size_t bytes,
                        std::stop_token stop,
                        clock::time_point deadline,
                        lease& out)
  {
    if (out || bytes == 0) return SIRIUS_INVALID_ARGUMENT;
    auto owner = state_;
    if (bytes > owner->capacity) return SIRIUS_RESOURCE_EXHAUSTED;
    std::unique_lock lock(owner->mutex);
    const auto fits = [&] {
      return owner->closed ||
             (owner->leases < owner->max_leases && bytes <= owner->capacity - owner->used);
    };
    if (!owner->available.wait_until(lock, stop, deadline, fits)) {
      return stop.stop_requested() ? SIRIUS_CANCELLED : SIRIUS_TIMEOUT;
    }
    if (stop.stop_requested() || owner->closed) return SIRIUS_CANCELLED;
    owner->used += bytes;
    ++owner->leases;
    if (owner->used > owner->peak) owner->peak = owner->used;
    out = lease(owner, bytes);
    return SIRIUS_OK;
  }

  void close() noexcept
  {
    std::shared_ptr<capacity_waker> waker;
    {
      std::lock_guard lock(state_->mutex);
      state_->closed = true;
      waker          = std::move(state_->waker);
    }
    state_->available.notify_all();
    if (waker) waker->wake();
  }

  void set_waker(std::shared_ptr<capacity_waker> waker)
  {
    {
      std::lock_guard lock(state_->mutex);
      if (state_->closed) throw std::logic_error("closed buffer budget");
      state_->waker.swap(waker);
    }
    // Destroy the previous callback outside the accounting mutex.
  }

  snapshot inspect() const
  {
    std::lock_guard lock(state_->mutex);
    return {state_->used, state_->leases, state_->peak, state_->closed};
  }

 private:
  std::shared_ptr<state> state_;
};
}  // namespace sirius::embedding
