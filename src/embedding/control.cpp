/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding/control.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <new>
#include <utility>

namespace sirius::embedding {
namespace {
sirius_error error(sirius_status code, const char* message = "") noexcept
{
  sirius_error result{};
  assign_error(result, code, message);
  return result;
}
bool quiesced(query_state const& q)
{
  return q.phase == query_phase::QUIESCED || q.phase == query_phase::CLOSED;
}
bool terminal(query_state const& q) { return quiesced(q) || q.phase == query_phase::UNAVAILABLE; }
}  // namespace
void assign_error(sirius_error& out, sirius_status code, const char* message) noexcept
{
  out.code = code;
  std::snprintf(out.message, sizeof(out.message), "%s", message ? message : "");
}
sirius_error current_error() noexcept
{
  try {
    throw;
  } catch (failure const& e) {
    return e.error;
  } catch (std::bad_alloc const&) {
    return error(SIRIUS_RESOURCE_EXHAUSTED, "native allocation failed");
  } catch (std::exception const& e) {
    return error(SIRIUS_EXECUTION_FAILED, e.what());
  } catch (...) {
    return error(SIRIUS_EXECUTION_FAILED, "unknown native failure");
  }
}

engine_control::engine_control(factory create_backend, std::size_t max_waiting)
  : create_backend_(std::move(create_backend)), max_waiting_(max_waiting)
{
  if (!create_backend_ || max_waiting == 0 || max_waiting > 16)
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid native coordinator configuration");
}
engine_control::~engine_control() { assert(!worker_.joinable()); }
sirius_error engine_control::initialize()
{
  worker_ = std::thread([this] { worker(); });
  std::unique_lock lock(mutex_);
  changed_.wait(lock, [&] { return initialized_; });
  auto result = initialization_error_;
  lock.unlock();
  if (result.code != SIRIUS_OK) worker_.join();
  return result;
}
std::shared_ptr<query_state> engine_control::create(std::string_view plan,
                                                    std::chrono::milliseconds timeout)
{
  if (plan.empty() || plan.size() > (16u << 20) || timeout.count() <= 0)
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid query plan or deadline");
  std::lock_guard lock(mutex_);
  if (unavailable_) throw failure(SIRIUS_GPU_UNAVAILABLE, "native runtime is unavailable");
  if (!initialized_ || initialization_error_.code != SIRIUS_OK)
    throw failure(SIRIUS_INVALID_STATE, "native runtime is not initialized");
  if (!accepting_) throw failure(SIRIUS_INVALID_STATE, "native runtime is stopping");
  if (live_ == max_waiting_ + 1)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native query capacity reached");
  // Admission precedes the bounded copy; rejected concurrent callers do not
  // each allocate a maximum-size plan outside the query-count limit.
  auto q      = std::make_shared<query_state>();
  q->plan     = plan;
  q->deadline = clock::now() + timeout;
  for (std::size_t i = 0; i < queries_.size(); ++i) {
    if (queries_[i].expired()) {
      q->slot     = i;
      queries_[i] = q;
      ++live_;
      return q;
    }
  }
  throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native query handles exhausted");
}
sirius_error engine_control::prepare(std::shared_ptr<query_state> const& q,
                                     std::chrono::milliseconds duration)
{
  std::unique_lock lock(mutex_);
  if (q->phase == query_phase::PREPARED || terminal(*q)) return q->result;
  if (q->phase != query_phase::CREATED && q->phase != query_phase::QUEUED &&
      q->phase != query_phase::PREPARING)
    return error(SIRIUS_INVALID_STATE, "query preparation already completed");
  if (q->phase == query_phase::CREATED) {
    if (!accepting_) return error(SIRIUS_INVALID_STATE, "native runtime is stopping");
    if (pending_.size() >= max_waiting_ + (active_ ? 0 : 1))
      return error(SIRIUS_RESOURCE_EXHAUSTED, "native preparation queue is full");
    pending_.push_back(q);
    q->phase = query_phase::QUEUED;
    changed_.notify_all();
  }
  if (!changed_.wait_for(
        lock, duration, [&] { return q->phase == query_phase::PREPARED || terminal(*q); }))
    return error(SIRIUS_TIMEOUT, "waiting for native preparation timed out");
  return q->result;
}
sirius_error engine_control::start(std::shared_ptr<query_state> const& q)
{
  std::lock_guard lock(mutex_);
  if (q->phase != query_phase::PREPARED || q->started || q->stop.stop_requested())
    return error(SIRIUS_INVALID_STATE, "query is not startable");
  q->started = true;
  changed_.notify_all();
  return {};
}
void engine_control::cancel(std::shared_ptr<query_state> const& q)
{
  // stop callbacks must be nonblocking; invoking one under mutex_ could deadlock.
  q->stop.request_stop();
  {
    std::lock_guard lock(mutex_);
    if (q->phase == query_phase::CREATED || q->phase == query_phase::QUEUED) {
      std::erase(pending_, q);
      q->result = error(SIRIUS_CANCELLED, "native query cancelled before preparation");
      q->phase  = query_phase::QUIESCED;
    }
  }
  changed_.notify_all();
}
sirius_error engine_control::wait(std::shared_ptr<query_state> const& q,
                                  std::chrono::milliseconds duration)
{
  std::unique_lock lock(mutex_);
  if (!changed_.wait_for(lock, duration, [&] { return terminal(*q); }))
    return error(SIRIUS_TIMEOUT, "waiting for native quiescence timed out");
  return q->result;
}
sirius_error engine_control::close_query(std::shared_ptr<query_state> const& q,
                                         std::chrono::milliseconds duration)
{
  cancel(q);
  auto result = wait(q, duration);
  // TIMEOUT may also be the execution outcome. Inspect the state rather than
  // confusing a quiesced expired query with a still-running wait timeout.
  std::lock_guard lock(mutex_);
  if (!quiesced(*q) || result.code == SIRIUS_GPU_UNAVAILABLE) return result;
  if (q->phase != query_phase::CLOSED) {
    q->phase = query_phase::CLOSED;
    queries_[q->slot].reset();
    --live_;
  }
  return {};
}
void engine_control::stop()
{
  std::array<std::shared_ptr<query_state>, 17> queries;
  {
    std::lock_guard lock(mutex_);
    accepting_ = false;
    for (std::size_t i = 0; i < queries_.size(); ++i)
      queries[i] = queries_[i].lock();
  }
  for (auto const& q : queries)
    if (q) cancel(q);
  changed_.notify_all();
}
sirius_error engine_control::close(std::chrono::milliseconds duration)
{
  stop();
  std::unique_lock lock(mutex_);
  if (unavailable_) return error(SIRIUS_GPU_UNAVAILABLE, "native runtime requires process restart");
  if (live_ != 0) return error(SIRIUS_BUSY, "close outstanding native query handles first");
  exit_requested_ = true;
  changed_.notify_all();
  if (!changed_.wait_for(lock, duration, [&] { return exited_; }))
    return error(SIRIUS_TIMEOUT, "native runtime shutdown timed out");
  lock.unlock();
  if (worker_.joinable()) worker_.join();
  return {};
}
sirius_engine_stats engine_control::inspect()
{
  std::lock_guard lock(mutex_);
  return {sizeof(sirius_engine_stats),
          SIRIUS_ABI_VERSION,
          accepting_,
          unavailable_,
          live_,
          pending_.size()};
}
void engine_control::process(engine_backend& backend,
                             std::shared_ptr<query_state> const& q) noexcept
{
  std::unique_ptr<query_driver> driver;
  sirius_error result{};
  try {
    if (q->stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "native query cancelled");
    if (clock::now() >= q->deadline) throw failure(SIRIUS_TIMEOUT, "native query deadline expired");
    driver = backend.prepare(q->plan, q->stop.get_token(), q->deadline);
    if (!driver) throw failure(SIRIUS_EXECUTION_FAILED, "native backend returned no query owner");
    {
      std::unique_lock lock(mutex_);
      q->phase = query_phase::PREPARED;
      changed_.notify_all();
      if (!changed_.wait_until(
            lock, q->deadline, [&] { return q->started || q->stop.stop_requested(); }) ||
          clock::now() >= q->deadline)
        throw failure(SIRIUS_TIMEOUT, "unstarted native query expired");
      if (q->stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "native query cancelled");
      q->phase = query_phase::RUNNING;
    }
    driver->run(q->stop.get_token(), q->deadline);
    if (q->stop.stop_requested())
      result = error(SIRIUS_CANCELLED, "native query cancelled");
    else if (clock::now() >= q->deadline)
      result = error(SIRIUS_TIMEOUT, "native query deadline expired");
  } catch (...) {
    result = current_error();
  }
  {
    std::lock_guard lock(mutex_);
    q->phase = query_phase::DRAINING;
  }
  if (driver && result.code != SIRIUS_GPU_UNAVAILABLE) {
    try {
      driver->finish();
    } catch (...) {
      result = error(SIRIUS_GPU_UNAVAILABLE, "native cleanup could not prove quiescence");
    }
  }
  if (result.code == SIRIUS_GPU_UNAVAILABLE) {
    // The process is now the cleanup owner. Do not destruct live CUDA buffers
    // or a thread-affine window after a failed synchronization.
    (void)driver.release();
  } else {
    driver.reset();
  }
  {
    std::lock_guard lock(mutex_);
    q->result = result;
    q->phase =
      result.code == SIRIUS_GPU_UNAVAILABLE ? query_phase::UNAVAILABLE : query_phase::QUIESCED;
    active_.reset();
    if (result.code == SIRIUS_GPU_UNAVAILABLE) {
      unavailable_ = true;
      accepting_   = false;
    }
  }
  if (result.code == SIRIUS_GPU_UNAVAILABLE) stop();
  changed_.notify_all();
}
void engine_control::worker() noexcept
{
  std::unique_ptr<engine_backend> backend;
  try {
    backend = create_backend_();
    if (!backend) throw failure(SIRIUS_EXECUTION_FAILED, "native backend initialization failed");
  } catch (...) {
    std::lock_guard lock(mutex_);
    initialization_error_ = current_error();
    initialized_ = exited_ = true;
    accepting_             = false;
    changed_.notify_all();
    return;
  }
  {
    std::lock_guard lock(mutex_);
    initialized_ = true;
    changed_.notify_all();
  }
  for (;;) {
    std::shared_ptr<query_state> q;
    {
      std::unique_lock lock(mutex_);
      changed_.wait(lock, [&] { return exit_requested_ || !pending_.empty(); });
      if (exit_requested_) break;
      q = std::move(pending_.front());
      pending_.pop_front();
      q->phase = query_phase::PREPARING;
      active_  = q;
    }
    process(*backend, q);
  }
  backend.reset();
  std::lock_guard lock(mutex_);
  exited_ = true;
  changed_.notify_all();
}
}  // namespace sirius::embedding
