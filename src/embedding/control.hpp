/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sirius_c.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

namespace sirius::embedding {
class native_input;
struct input_registry;
using clock = std::chrono::steady_clock;
void assign_error(sirius_error& out, sirius_status code, const char* message) noexcept;
sirius_error current_error() noexcept;

class failure : public std::exception {
 public:
  failure(sirius_status code, const char* message) noexcept { assign_error(error, code, message); }
  const char* what() const noexcept override { return error.message; }
  sirius_error error{};
};

// All methods, including destruction, run on the coordinator. run must close
// its cancellation subscriptions and drain tasks before returning or throwing.
// A GPU_UNAVAILABLE exception means destruction is unsafe: retain the driver.
class query_driver {
 public:
  virtual ~query_driver()                                            = default;
  virtual void run(std::stop_token stop, clock::time_point deadline) = 0;
  virtual void finish()                                              = 0;
};

class engine_backend {
 public:
  virtual ~engine_backend() = default;
  virtual std::unique_ptr<query_driver> prepare_inputs(std::string_view plan,
                                                       input_registry& inputs,
                                                       std::stop_token stop,
                                                       clock::time_point deadline)
  {
    return prepare(plan, stop, deadline);
  }
  virtual std::unique_ptr<query_driver> prepare(std::string_view plan,
                                                std::stop_token stop,
                                                clock::time_point deadline) = 0;
};

enum class query_phase {
  CREATED,
  QUEUED,
  PREPARING,
  PREPARED,
  RUNNING,
  DRAINING,
  QUIESCED,
  UNAVAILABLE,
  CLOSED
};
struct query_state {
  std::shared_ptr<input_registry> inputs;
  std::string plan;
  clock::time_point deadline;
  std::stop_source stop;
  query_phase phase{query_phase::CREATED};
  bool started{false};
  sirius_error result{};
  std::size_t slot{0};
};

// Bounded single-owner coordinator, independent of DuckDB/CUDA for deterministic
// control tests. No worker-pool joins or engine locks on the cancellation path.
class engine_control {
 public:
  using factory = std::function<std::unique_ptr<engine_backend>()>;
  engine_control(factory create_backend, std::size_t max_waiting);
  ~engine_control();
  engine_control(engine_control const&)            = delete;
  engine_control& operator=(engine_control const&) = delete;

  sirius_error initialize();
  std::shared_ptr<query_state> create(std::string_view plan, std::chrono::milliseconds timeout);
  sirius_error prepare(std::shared_ptr<query_state> const& q, std::chrono::milliseconds wait);
  sirius_error start(std::shared_ptr<query_state> const& q);
  void cancel(std::shared_ptr<query_state> const& q);
  sirius_error wait(std::shared_ptr<query_state> const& q, std::chrono::milliseconds duration);
  sirius_error close_query(std::shared_ptr<query_state> const& q,
                           std::chrono::milliseconds duration);
  void stop();
  sirius_error close(std::chrono::milliseconds duration);
  sirius_engine_stats inspect();
  void require_input_active(std::shared_ptr<query_state> const& q);
  std::shared_ptr<native_input> register_input(std::shared_ptr<query_state> const& q,
                                               uint64_t id,
                                               const sirius_input_column* columns,
                                               uint32_t count);

 private:
  void worker() noexcept;
  void process(engine_backend& backend, std::shared_ptr<query_state> const& q) noexcept;
  factory create_backend_;
  const std::size_t max_waiting_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::thread worker_;
  std::array<std::weak_ptr<query_state>, 17> queries_;
  std::deque<std::shared_ptr<query_state>> pending_;
  std::shared_ptr<query_state> active_;
  std::size_t live_{0};
  bool initialized_{false}, accepting_{true}, unavailable_{false};
  bool exit_requested_{false}, exited_{false};
  sirius_error initialization_error_{};
};

engine_control::factory native_backend_factory(std::string config_path, uint32_t gpu_streams);
}  // namespace sirius::embedding
