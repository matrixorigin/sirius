/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "catch.hpp"
#include "exec/config.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "scan/test_utils.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

using namespace sirius::pipeline;
using namespace sirius::parallel;
using namespace std::chrono_literals;
using namespace sirius::op;

/**
 * Mock GPU pipeline task for testing.
 * This task simulates work without actually executing GPU operations.
 */
class mock_gpu_pipeline_task_global_state : public gpu_pipeline_task_global_state {
 public:
  mock_gpu_pipeline_task_global_state()
    : gpu_pipeline_task_global_state(nullptr), executed_count(0), gpu_ids_used()
  {
  }

  std::atomic<int> executed_count;
  std::vector<int> gpu_ids_used;
  std::mutex gpu_ids_mutex;
};

class mock_gpu_pipeline_task_local_state : public gpu_pipeline_task_local_state {
 public:
  mock_gpu_pipeline_task_local_state(int task_id, int expected_gpu_id)
    : gpu_pipeline_task_local_state(std::make_unique<pipelineable_operator_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{})),
      _task_id(task_id),
      _expected_gpu_id(expected_gpu_id)
  {
  }

  int _task_id;
  int _expected_gpu_id;
};

class mock_gpu_pipeline_task : public gpu_pipeline_task {
 public:
  mock_gpu_pipeline_task(std::unique_ptr<mock_gpu_pipeline_task_local_state> local_state,
                         std::shared_ptr<mock_gpu_pipeline_task_global_state> global_state)
    : gpu_pipeline_task(0,
                        std::vector<cucascade::shared_data_repository*>{},
                        std::move(local_state),
                        std::move(global_state))
  {
  }

  void execute(rmm::cuda_stream_view stream) override
  {
    auto& global = _global_state->cast<mock_gpu_pipeline_task_global_state>();
    auto& local  = _local_state->cast<mock_gpu_pipeline_task_local_state>();

    // Simulate some work
    std::this_thread::sleep_for(5ms);

    // Increment counter
    global.executed_count.fetch_add(1, std::memory_order_relaxed);

    // Record which GPU (thread) executed this task
    {
      std::lock_guard<std::mutex> lock(global.gpu_ids_mutex);
      global.gpu_ids_used.push_back(local._task_id);
    }
  }
};

struct cross_pipeline_overlap {
  std::mutex mutex;
  std::condition_variable cv;
  int active{0};
  int max_active{0};
};

class overlap_tracking_operator : public sirius_physical_operator {
 public:
  explicit overlap_tracking_operator(std::shared_ptr<cross_pipeline_overlap> cross = nullptr)
    : sirius_physical_operator(SiriusPhysicalOperatorType::PROJECTION, {}, 0),
      cross(std::move(cross))
  {
  }

  std::unique_ptr<operator_data> execute(const operator_data&, rmm::cuda_stream_view) override
  {
    auto current  = active.fetch_add(1) + 1;
    auto observed = max_active.load();
    while (observed < current && !max_active.compare_exchange_weak(observed, current)) {}
    if (cross) {
      std::unique_lock lock(cross->mutex);
      cross->active++;
      cross->max_active = std::max(cross->max_active, cross->active);
      cross->cv.notify_all();
      cross->cv.wait_for(lock, 1s, [&] { return cross->max_active >= 2; });
      cross->active--;
    } else {
      std::this_thread::sleep_for(50ms);
    }
    active.fetch_sub(1);
    executed.fetch_add(1);
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  std::atomic<int> active{0};
  std::atomic<int> max_active{0};
  std::atomic<int> executed{0};
  std::shared_ptr<cross_pipeline_overlap> cross;
};

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_small_test_memory_manager()
{
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(256 * 1024 * 1024)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_host_capacity(256 * 1024 * 1024)
    .use_host_per_gpu()
    .track_reservation_per_stream(false)
    .set_reservation_fraction_per_host(0.75);
  return std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
}

TEST_CASE("Task scheduler can start and stop gracefully", "[task_scheduler]")
{
  auto manager = make_small_test_memory_manager();
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{2};
  task_scheduler executor(gpu_config, scan_config, *manager);

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("Task scheduler executes tasks through pipeline_queue", "[task_scheduler]")
{
  auto manager = make_small_test_memory_manager();
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{2};
  task_scheduler executor(gpu_config, scan_config, *manager);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Schedule multiple tasks
  const int num_tasks = 10;
  for (int i = 0; i < num_tasks; ++i) {
    auto local_state = std::make_unique<mock_gpu_pipeline_task_local_state>(i, 0);
    auto task = std::make_unique<mock_gpu_pipeline_task>(std::move(local_state), global_state);
    executor.schedule(std::move(task));
  }

  // Wait for all tasks to complete
  auto start_time = std::chrono::steady_clock::now();
  auto timeout    = std::chrono::seconds(10);
  while (global_state->executed_count.load(std::memory_order_relaxed) < num_tasks) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out waiting for tasks to complete");
    }
  }

  REQUIRE(global_state->executed_count.load() == num_tasks);

  executor.stop();
}

TEST_CASE("GPU executor serializes complete tasks within one pipeline",
          "[task_scheduler][pipeline_execution_gate]")
{
  auto manager = make_small_test_memory_manager();
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{1};
  task_scheduler executor(gpu_config, scan_config, *manager);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  sirius::sirius_interface sirius_iface(*con.context);
  sirius::sirius_engine engine(*con.context, sirius_iface);
  auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(engine);
  overlap_tracking_operator op;
  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*pipeline, op);
  build_state.add_pipeline_operator(*pipeline, op);
  build_state.set_pipeline_sink(*pipeline, &op, 0);
  op.set_pipeline(pipeline);

  auto global_state = std::make_shared<gpu_pipeline_task_global_state>(pipeline);
  pipeline->mark_task_created();
  pipeline->mark_task_created();
  executor.start();
  for (uint64_t task_id = 0; task_id < 2; ++task_id) {
    auto input = std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
    auto local_state = std::make_unique<gpu_pipeline_task_local_state>(std::move(input));
    executor.schedule(
      std::make_unique<gpu_pipeline_task>(task_id,
                                          std::vector<cucascade::shared_data_repository*>{},
                                          std::move(local_state),
                                          global_state));
  }

  auto const deadline = std::chrono::steady_clock::now() + 10s;
  while (op.executed.load() < 2 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  executor.stop();

  REQUIRE(op.executed.load() == 2);
  CHECK(op.max_active.load() == 1);
}

TEST_CASE("GPU executor runs independent pipelines on separate streams",
          "[task_scheduler][pipeline_execution_gate]")
{
  auto manager = make_small_test_memory_manager();
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{1};
  task_scheduler executor(gpu_config, scan_config, *manager);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  sirius::sirius_interface sirius_iface(*con.context);
  sirius::sirius_engine engine(*con.context, sirius_iface);
  auto cross = std::make_shared<cross_pipeline_overlap>();

  auto make_pipeline = [&](overlap_tracking_operator& op) {
    auto pipeline = duckdb::make_shared_ptr<sirius_pipeline>(engine);
    sirius_pipeline_build_state build_state;
    build_state.set_pipeline_source(*pipeline, op);
    build_state.add_pipeline_operator(*pipeline, op);
    build_state.set_pipeline_sink(*pipeline, &op, 0);
    op.set_pipeline(pipeline);
    pipeline->mark_task_created();
    return pipeline;
  };

  overlap_tracking_operator first_op(cross);
  overlap_tracking_operator second_op(cross);
  auto first_pipeline  = make_pipeline(first_op);
  auto second_pipeline = make_pipeline(second_op);

  executor.start();
  auto schedule_pipeline = [&](uint64_t task_id,
                               const duckdb::shared_ptr<sirius_pipeline>& pipeline) {
    auto global_state = std::make_shared<gpu_pipeline_task_global_state>(pipeline);
    auto input        = std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
    auto local_state = std::make_unique<gpu_pipeline_task_local_state>(std::move(input));
    executor.schedule(
      std::make_unique<gpu_pipeline_task>(task_id,
                                          std::vector<cucascade::shared_data_repository*>{},
                                          std::move(local_state),
                                          std::move(global_state)));
  };
  schedule_pipeline(0, first_pipeline);
  schedule_pipeline(1, second_pipeline);

  auto const deadline = std::chrono::steady_clock::now() + 10s;
  while ((first_op.executed.load() + second_op.executed.load()) < 2 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  executor.stop();

  REQUIRE(first_op.executed.load() == 1);
  REQUIRE(second_op.executed.load() == 1);
  CHECK(cross->max_active == 2);
}

TEST_CASE("Task queue handles empty queue gracefully", "[pipeline_queue]")
{
  auto manager = make_small_test_memory_manager();
  sirius::exec::thread_pool_config gpu_config{2};
  sirius::exec::thread_pool_config scan_config{2};
  task_scheduler executor(gpu_config, scan_config, *manager);

  auto global_state = std::make_shared<mock_gpu_pipeline_task_global_state>();

  executor.start();

  // Don't schedule any tasks, just verify clean shutdown
  std::this_thread::sleep_for(50ms);

  REQUIRE(global_state->executed_count.load() == 0);

  REQUIRE_NOTHROW(executor.stop());
}
