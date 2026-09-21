/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/control.hpp"
#include "embedding/input.hpp"
#include "embedding/result.hpp"
#include "embedding/tae_demand.hpp"

#include <catch.hpp>

#include <array>
#include <atomic>
#include <exception>
#include <future>
#include <latch>
#include <thread>
#include <vector>

using namespace sirius::embedding;
using namespace std::chrono_literals;

namespace {
struct stats_storage final : input_storage {
  std::vector<std::byte> bytes;
  explicit stats_storage(std::size_t size) : bytes(size) {}
  std::size_t size() const override { return bytes.size(); }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& fn) override
  {
    fn(0, bytes);
  }
};
struct stats_pool final : input_pool {
  std::size_t rounded(std::size_t bytes) const override { return (bytes + 7) / 8 * 8; }
  std::unique_ptr<input_storage> allocate(std::size_t bytes) override
  {
    return std::make_unique<stats_storage>(rounded(bytes));
  }
};
struct charged_pool final : input_pool {
  std::size_t rounded(std::size_t bytes) const override { return bytes; }
  std::unique_ptr<input_storage> allocate(std::size_t) override
  {
    // The handoff test exercises accounting, not a 64 MiB payload.
    return std::make_unique<stats_storage>(0);
  }
};
struct callback_waker final : capacity_waker {
  std::function<void()> callback;
  std::exception_ptr error;
  void wake() noexcept override
  {
    try {
      callback();
    } catch (...) {
      error = std::current_exception();
    }
  }
};
class stats_driver final : public query_driver {
 public:
  void run(std::stop_token, clock::time_point) override {}
  void finish() override {}
};
class stats_backend final : public engine_backend {
 public:
  std::unique_ptr<query_driver> prepare(std::string_view,
                                        std::stop_token,
                                        clock::time_point) override
  {
    return std::make_unique<stats_driver>();
  }
};
}  // namespace

TEST_CASE("native execution stats retain query terminal state until close", "[native_stats][cpu]")
{
  engine_control control([] { return std::make_unique<stats_backend>(); }, 1);
  REQUIRE(control.initialize().code == SIRIUS_OK);
  auto query = control.create("stats", 5s);
  CHECK_FALSE(control.inspect_execution(query).terminal);
  REQUIRE(control.prepare(query, 5s).code == SIRIUS_OK);
  REQUIRE(control.start(query).code == SIRIUS_OK);
  REQUIRE(control.wait(query, 5s).code == SIRIUS_OK);
  auto snapshot = control.inspect_execution(query);
  CHECK(snapshot.terminal);
  CHECK(snapshot.terminal_status == SIRIUS_OK);
  CHECK_FALSE(snapshot.fatal);
  REQUIRE(control.close_query(query, 5s).code == SIRIUS_OK);
  REQUIRE(control.close(5s).code == SIRIUS_OK);
}

TEST_CASE("native execution stats publish one untorn first terminal outcome", "[native_stats][cpu]")
{
  execution_stats sequential;
  sequential.terminal(SIRIUS_EXECUTION_FAILED, false);
  sequential.terminal(SIRIUS_GPU_UNAVAILABLE, true);
  auto snapshot = sequential.inspect();
  CHECK(snapshot.terminal);
  CHECK(snapshot.terminal_status == SIRIUS_EXECUTION_FAILED);
  CHECK_FALSE(snapshot.fatal);

  execution_stats concurrent;
  std::latch start(1);
  std::thread ordinary([&] {
    start.wait();
    concurrent.terminal(SIRIUS_CANCELLED, false);
  });
  std::thread fatal([&] {
    start.wait();
    concurrent.terminal(SIRIUS_GPU_UNAVAILABLE, true);
  });
  start.count_down();
  ordinary.join();
  fatal.join();
  snapshot = concurrent.inspect();
  REQUIRE(snapshot.terminal);
  CHECK(((snapshot.terminal_status == SIRIUS_CANCELLED && !snapshot.fatal) ||
         (snapshot.terminal_status == SIRIUS_GPU_UNAVAILABLE && snapshot.fatal)));
}

TEST_CASE("native execution stats aggregate charged MO and result ownership", "[native_stats][cpu]")
{
  auto stats = std::make_shared<execution_stats>();
  stats->add_source(SIRIUS_READ_MO);
  stats->tae_gpu_admission_wait();
  stats->tae_gpu_reservation(4096);
  auto mo_only = stats->inspect();
  CHECK(mo_only.tae_requests == 0);
  CHECK(mo_only.tae_work_issued == 0);
  CHECK(mo_only.tae_work_completed == 0);
  CHECK(mo_only.tae_active_work == 0);
  CHECK(mo_only.tae_peak_active_work == 0);
  CHECK(mo_only.tae_peak_queued_work == 0);
  CHECK(mo_only.tae_work_limit == 0);
  CHECK(mo_only.tae_slice_bytes == 0);
  CHECK(mo_only.tae_peak_cached_metadata_charged_bytes == 0);
  CHECK(mo_only.tae_peak_staging_charged_bytes == 0);
  CHECK(mo_only.tae_gpu_admission_waits == 0);
  CHECK(mo_only.tae_peak_gpu_reservation_admitted_bytes == 0);
  CHECK(mo_only.tae_payload_bytes == 0);
  auto pool  = std::make_shared<stats_pool>();
  auto input = std::make_shared<native_input>(1,
                                              std::vector<sirius_input_column>{{23, 0, 0, 0}},
                                              std::stop_token{},
                                              clock::now() + 5s,
                                              256,
                                              4,
                                              stats);
  input->activate(pool);
  auto batch = input->acquire(64, clock::now());
  sirius_input_vector vector{};
  vector.data_bytes = 64;
  input->publish(batch, 8, {&vector, 1});
  batch.reset();
  auto unit = input->claim();
  REQUIRE(unit);
  auto snapshot = stats->inspect();
  CHECK(snapshot.source_mask == SIRIUS_QUERY_SOURCE_MO);
  CHECK(snapshot.mo_input_units == 1);
  CHECK(snapshot.mo_input_retained_charged_bytes > 64);
  CHECK(snapshot.mo_input_peak_charged_bytes == snapshot.mo_input_retained_charged_bytes);
  unit.reset();
  CHECK(stats->inspect().mo_input_retained_charged_bytes == 0);
  input->stop();
  input->discard();

  auto result = std::make_shared<native_result>(stats);
  result->activate(pool);
  std::shared_ptr<result_batch> output;
  REQUIRE(result->try_allocate(128, 1, output) == SIRIUS_OK);
  output->rows = 2;
  result->publish(std::move(output));
  output = result->next(clock::now());
  std::shared_ptr<result_batch> blocked;
  REQUIRE(result->try_allocate(result_window - sizeof(result_batch), 0, blocked) == SIRIUS_TIMEOUT);
  result->parked(true);
  snapshot = stats->inspect();
  CHECK(snapshot.result_rows == 2);
  CHECK(snapshot.result_payload_bytes == 128);
  CHECK(snapshot.result_retained_charged_bytes > 128);
  CHECK(snapshot.result_peak_charged_bytes == snapshot.result_retained_charged_bytes);
  CHECK(snapshot.result_blocked_publications == 1);
  CHECK(snapshot.result_parked_publications == 1);
  result->parked(false);
  output.reset();
  CHECK(stats->inspect().result_retained_charged_bytes == 0);
  CHECK(stats->inspect().result_parked_publications == 0);
  result->cancel();
}

TEST_CASE("native execution stats retain bounded TAE demand activity", "[native_stats][cpu]")
{
  auto stats = std::make_shared<execution_stats>();
  stats->add_source(SIRIUS_READ_TAE);
  auto budget     = std::make_shared<buffer_budget>(8192, 2);
  auto controller = make_tae_demand_controller(1, budget, 8192, stats);
  std::promise<std::shared_ptr<tae_work_permit>> first, second, third;
  auto a = first.get_future();
  auto b = second.get_future();
  auto c = third.get_future();
  REQUIRE(controller->request([&](auto permit) { first.set_value(std::move(permit)); }));
  REQUIRE(controller->request([&](auto permit) { second.set_value(std::move(permit)); }));
  REQUIRE(a.wait_for(5s) == std::future_status::ready);
  REQUIRE(b.wait_for(5s) == std::future_status::ready);
  auto one = a.get();
  auto two = b.get();
  REQUIRE(one);
  REQUIRE(two);
  REQUIRE(controller->request([&](auto permit) { third.set_value(std::move(permit)); }));
  CHECK(c.wait_for(0s) == std::future_status::timeout);
  controller->cache_metadata("object", std::make_shared<const int>(42), 32);
  one->record_payload_bytes(17);
  auto snapshot = stats->inspect();
  CHECK(snapshot.tae_requests == 3);
  CHECK(snapshot.tae_work_issued == 2);
  CHECK(snapshot.tae_active_work == 2);
  CHECK(snapshot.tae_peak_active_work == 2);
  CHECK(snapshot.tae_peak_queued_work >= 1);
  CHECK(snapshot.tae_work_limit == 2);
  CHECK(snapshot.tae_slice_bytes == 4096);
  CHECK(snapshot.tae_peak_staging_charged_bytes == 8192);
  CHECK(snapshot.tae_peak_cached_metadata_charged_bytes > 32);
  CHECK(snapshot.tae_payload_bytes == 17);

  one.reset();
  REQUIRE(c.wait_for(5s) == std::future_status::ready);
  auto three = c.get();
  REQUIRE(three);
  two.reset();
  three.reset();
  controller->close();
  snapshot = stats->inspect();
  CHECK(snapshot.tae_work_issued == 3);
  CHECK(snapshot.tae_work_completed == 3);
  CHECK(snapshot.tae_active_work == 0);
}

TEST_CASE("native result stats retire charge before waking a capacity handoff",
          "[native_stats][cpu]")
{
  auto stats  = std::make_shared<execution_stats>();
  auto result = std::make_shared<native_result>(stats);
  result->activate(std::make_shared<charged_pool>());
  auto const payload = result_window - sizeof(result_batch);
  std::shared_ptr<result_batch> retiring;
  REQUIRE(result->try_allocate(payload, 0, retiring) == SIRIUS_OK);
  REQUIRE(stats->inspect().result_retained_charged_bytes == result_window);

  std::shared_ptr<result_batch> replacement;
  sirius_status handoff_status = SIRIUS_BUSY;
  auto wake                    = std::make_shared<callback_waker>();
  std::atomic<bool> fired{false};
  wake->callback =
    [weak = std::weak_ptr<native_result>(result), &replacement, &handoff_status, &fired] {
      if (fired.exchange(true)) return;
      auto owner     = weak.lock();
      handoff_status = owner
                         ? owner->try_allocate(result_window - sizeof(result_batch), 0, replacement)
                         : SIRIUS_INVALID_STATE;
    };
  result->subscribe(wake);
  retiring.reset();
  REQUIRE(fired.load());
  REQUIRE_FALSE(wake->error);
  REQUIRE(handoff_status == SIRIUS_OK);
  REQUIRE(replacement);
  auto snapshot = stats->inspect();
  CHECK(snapshot.result_retained_charged_bytes == result_window);
  CHECK(snapshot.result_peak_charged_bytes == result_window);
  replacement.reset();
  CHECK(stats->inspect().result_retained_charged_bytes == 0);
  result->cancel();
}

TEST_CASE("TAE demand admission failure does not publish a nonexistent permit",
          "[native_stats][cpu]")
{
  auto stats = std::make_shared<execution_stats>();
  stats->add_source(SIRIUS_READ_TAE);
  auto budget     = std::make_shared<buffer_budget>(8192, 2);
  auto controller = make_tae_demand_controller(1, budget, 8192, stats);
  budget->close();
  std::promise<bool> delivered;
  auto ready = delivered.get_future();
  REQUIRE(controller->request(
    [&](std::shared_ptr<tae_work_permit> permit) { delivered.set_value(permit != nullptr); }));
  REQUIRE(ready.wait_for(5s) == std::future_status::ready);
  CHECK_FALSE(ready.get());
  auto snapshot = stats->inspect();
  CHECK(snapshot.tae_requests == 1);
  CHECK(snapshot.tae_work_issued == 0);
  CHECK(snapshot.tae_work_completed == 0);
  CHECK(snapshot.tae_active_work == 0);
  CHECK(snapshot.tae_peak_active_work == 0);
  CHECK(snapshot.tae_peak_staging_charged_bytes == 0);
  CHECK(controller->inspect().active == 0);
  controller->close();
}
