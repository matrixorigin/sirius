/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "../utils/telemetry_utils.hpp"
#include "embedding/control.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"

#include <catch.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

namespace {
struct bounded_input : sirius::op::operator_data {
  std::size_t mandatory_gpu_reservation_bytes() const noexcept override { return 20u << 20; }
};
struct admission_task : sirius::pipeline::gpu_pipeline_task {
  std::size_t retry_floor{};
  admission_task(std::shared_ptr<sirius::pipeline::gpu_pipeline_task_global_state> state)
    : gpu_pipeline_task(1,
                        {},
                        std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(
                          std::make_unique<bounded_input>()),
                        std::move(state))
  {
  }
  sirius::pipeline::reservation_size_info get_estimated_reservation_size_info(
    const cucascade::memory::memory_space*) const override
  {
    sirius::pipeline::reservation_size_info info;
    info.mandatory_gpu_bytes     = 20u << 20;
    info.retry_reservation_floor = retry_floor;
    info.reservation_size        = std::max<std::size_t>(40u << 20, retry_floor);
    return info;
  }
};
}  // namespace

TEST_CASE("full converter admission parks before claiming GPU work and respects retry floors",
          "[native_admission][gpu]")
{
  using namespace sirius;
  cucascade::memory::reservation_manager_configurator builder;
  auto configs = builder.set_number_of_gpus(1)
                   .set_gpu_usage_limit(256u << 20)
                   .set_reservation_fraction_per_gpu(0.75)
                   .set_per_numa_region_capacity(128u << 20)
                   .use_gpu_id_as_host_id()
                   .track_reservation_per_stream(false)
                   .set_reservation_fraction_per_numa_region(0.75)
                   .build();
  memory::sirius_memory_reservation_manager manager(std::move(configs));
  auto* space = manager.get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  exec::channel<std::unique_ptr<pipeline::task_request>> channel;
  exec::thread_pool_config config;
  config.num_threads = 2;
  auto telemetry     = test::make_test_telemetry_context();
  pipeline::gpu_pipeline_executor executor(
    config, space, channel.make_publisher(), nullptr, telemetry);
  auto state      = std::make_shared<pipeline::gpu_pipeline_task_global_state>(nullptr, telemetry);
  auto completion = std::make_shared<pipeline::completion_handler>();
  state->set_completion_handler(completion);
  auto done = completion->get_awaitable();
  admission_task task(state);
  auto* local = dynamic_cast<pipeline::gpu_pipeline_task_local_state*>(task.local_state());
  REQUIRE(local);
  auto occupied = space->make_reservation_or_null(space->get_max_memory());
  REQUIRE(occupied);
  CHECK_FALSE(executor.try_admit(task));
  CHECK(local->reservation() == nullptr);
  CHECK_FALSE(completion->has_error());
  occupied.reset();
  REQUIRE(executor.try_admit(task));
  REQUIRE(local->reservation());
  CHECK(local->reservation()->size() >= (20u << 20));
  local->release_reservation().reset();
  task.retry_floor = 60u << 20;
  occupied         = space->make_reservation_or_null(space->get_max_memory() - (30u << 20));
  REQUIRE(occupied);
  CHECK_FALSE(executor.try_admit(task));
  CHECK(local->reservation() == nullptr);
  occupied.reset();
  REQUIRE(executor.try_admit(task));
  REQUIRE(local->reservation());
  CHECK(local->reservation()->size() >= task.retry_floor);
  local->release_reservation().reset();
  task.retry_floor = space->get_max_memory() + 1;
  CHECK(executor.try_admit(task));  // Remove the failed task instead of waiting forever.
  CHECK(completion->has_error());
  CHECK(local->reservation() == nullptr);
  REQUIRE_THROWS_AS(done.get(), embedding::failure);
}
