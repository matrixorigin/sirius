/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "embedding/buffer_budget.hpp"
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
#include <vector>

namespace sirius::embedding {
class native_input;
class native_result;
struct input_registry;
struct query_state;
struct owned_column {
  uint32_t oid{};
  int32_t width{}, scale{};
  bool nullable{};
  std::string name;
};
struct owned_read_column {
  owned_column logical;
  uint64_t physical_column_id{};
  uint32_t sequence_number{};
};
struct owned_query_contract {
  uint32_t account_id{};
  std::string query_id;
  std::array<uint8_t, 12> snapshot_ts{};
  std::vector<owned_column> outputs;
};
struct owned_read_binding {
  uint64_t binding_id{};
  uint32_t source_kind{};
  std::string database_name, table_name, schema_name, manifest, data_root;
  std::vector<owned_read_column> columns;
};
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
  virtual bool startable() const noexcept { return true; }
};

class engine_backend {
 public:
  virtual ~engine_backend() = default;
  virtual bool available() const noexcept { return true; }
  virtual std::size_t metadata_capacity_bytes() const noexcept { return 256u << 20; }
  virtual std::unique_ptr<query_driver> prepare_inputs(std::string_view plan,
                                                       input_registry& inputs,
                                                       std::stop_token stop,
                                                       clock::time_point deadline)
  {
    return prepare(plan, stop, deadline);
  }
  virtual std::unique_ptr<query_driver> prepare_bound(std::string_view plan,
                                                      query_state const&,
                                                      input_registry& inputs,
                                                      std::stop_token stop,
                                                      clock::time_point deadline)
  {
    return prepare_inputs(plan, inputs, stop, deadline);
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
  std::shared_ptr<native_result> results;
  std::vector<sirius_column> result_schema;
  std::string plan;
  clock::time_point deadline;
  std::stop_source stop;
  query_phase phase{query_phase::CREATED};
  bool started{false};
  sirius_error result{};
  std::size_t slot{0};
  std::size_t metadata_bytes{0};
  std::vector<buffer_budget::lease> metadata_leases;
  bool startable{false};
  std::unique_ptr<owned_query_contract> contract;
  std::vector<owned_read_binding> bindings;
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
  void bind_query(std::shared_ptr<query_state> const&, const sirius_query_contract&);
  void register_read(std::shared_ptr<query_state> const&, const sirius_read_binding&);
  sirius_result_schema result_schema(std::shared_ptr<query_state> const&);
  void require_result_active(std::shared_ptr<query_state> const&);

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
  std::unique_ptr<buffer_budget> metadata_budget_;
};

void validate_embedded_plan(std::string_view plan, query_state const& query);

engine_control::factory native_backend_factory(std::string config_path, uint32_t gpu_streams);
}  // namespace sirius::embedding
