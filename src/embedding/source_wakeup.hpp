/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "embedding/buffer_budget.hpp"

#include <functional>

namespace sirius::embedding {
// One durable notification per source. dispatch only enqueues; it must not
// synchronously call complete(). close synchronizes with enqueue callbacks,
// then the query owner drains queued/in-flight requests before freeing a plan.
class source_wakeup final : public capacity_waker {
 public:
  source_wakeup(std::function<void()> dispatch, std::function<void(std::exception_ptr)> failed)
    : dispatch_(std::move(dispatch)), failed_(std::move(failed))
  {
  }
  void wake() noexcept override
  {
    std::lock_guard lock(mutex_);
    if (closed_) return;
    dirty_ = true;
    dispatch_locked();
  }
  void complete() noexcept
  {
    std::lock_guard lock(mutex_);
    pending_ = false;
    dispatch_locked();
  }
  void close() noexcept
  {
    std::lock_guard lock(mutex_);
    closed_ = true;
    dirty_  = false;
  }

 private:
  void dispatch_locked() noexcept
  {
    if (closed_ || pending_ || !dirty_) return;
    pending_ = true;
    dirty_   = false;
    try {
      dispatch_();
    } catch (...) {
      closed_ = true;
      failed_(std::current_exception());
    }
  }
  std::mutex mutex_;
  bool dirty_{false}, pending_{false}, closed_{false};
  std::function<void()> dispatch_;
  std::function<void(std::exception_ptr)> failed_;
};
}  // namespace sirius::embedding
