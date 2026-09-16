/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/input.hpp"
#include "embedding/source_wakeup.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <catch.hpp>

#include <cstring>
#include <future>
#include <latch>
using namespace sirius::embedding;
using namespace std::chrono_literals;
namespace {
struct heap_storage : input_storage {
  std::vector<std::byte> data;
  explicit heap_storage(std::size_t n) : data(n) {}
  std::size_t size() const override { return data.size(); }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& fn) override
  {
    fn(0, data);
  }
};
struct heap_pool : input_pool {
  std::size_t rounded(std::size_t n) const override { return (n + 7) / 8 * 8; }
  std::unique_ptr<input_storage> allocate(std::size_t n) override
  {
    return std::make_unique<heap_storage>(n);
  }
};
struct input_fixture {
  std::stop_source stop;
  std::shared_ptr<native_input> input;
  explicit input_fixture(sirius_input_column column = {23, 0, 0, 1}, std::size_t capacity = 128)
    : input(std::make_shared<native_input>(
        1, std::vector{column}, stop.get_token(), clock::now() + 10s, capacity, 4))
  {
    input->activate(std::make_shared<heap_pool>());
  }
  ~input_fixture()
  {
    input->stop();
    input->discard();
  }
};
struct waker : capacity_waker {
  std::atomic<int> calls{0};
  void wake() noexcept override { ++calls; }
};
}  // namespace
TEST_CASE("native input credits follow filling queued and retry owners", "[native_input]")
{
  input_fixture f;
  auto batch = f.input->acquire(64, clock::now());
  int64_t values[]{1, 2, 3, 4, 5, 6, 7, 8};
  batch->write(0, {reinterpret_cast<const std::byte*>(values), sizeof(values)});
  sirius_input_vector column{};
  column.data_bytes = 64;
  f.input->publish(batch, 8, {&column, 1});
  batch.reset();
  CHECK(f.input->inspect().retained_bytes == 64 + sizeof(sirius_input_vector));
  CHECK(f.input->inspect().queued_batches == 1);
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 8);
  CHECK(f.input->inspect().queued_batches == 0);
  CHECK(f.input->inspect().retained_bytes == 64 + sizeof(sirius_input_vector));
  REQUIRE_THROWS_AS(f.input->acquire(8, clock::now()), failure);
  auto retry = unit->slices[0].batch;
  unit.reset();
  CHECK(f.input->inspect().retained_bytes == 64 + sizeof(sirius_input_vector));
  retry.reset();
  CHECK(f.input->inspect().retained_bytes == 0);
  CHECK(f.input->inspect().source_units == 0);
}
TEST_CASE("native full input acquire cancels without invalidating filling storage",
          "[native_input]")
{
  input_fixture f;
  auto held = f.input->acquire(64, clock::now());
  std::latch entered(1);
  auto waiter = std::async(std::launch::async, [&]() -> int {
    entered.count_down();
    try {
      (void)f.input->acquire(8, clock::now() + 5s);
      return SIRIUS_OK;
    } catch (failure const& e) {
      return static_cast<int>(e.error.code);
    }
  });
  entered.wait();
  f.stop.request_stop();
  REQUIRE(waiter.get() == SIRIUS_CANCELLED);
  std::byte value{42};
  held->write(0, {&value, 1});
  CHECK(f.input->inspect().retained_bytes == 64 + sizeof(sirius_input_vector));
  held.reset();
  CHECK(f.input->inspect().retained_bytes == 0);
}
TEST_CASE("native EOS and publication ownership are explicit", "[native_input]")
{
  input_fixture f;
  REQUIRE_FALSE(f.input->exhausted());
  REQUIRE_FALSE(f.input->claim());
  auto held = f.input->acquire(8, clock::now());
  REQUIRE_THROWS_AS(f.input->finish(), failure);
  sirius_input_vector bad{};
  bad.data_bytes = 8;
  REQUIRE_THROWS_AS(f.input->publish(held, 2, {&bad, 1}), failure);
  CHECK_FALSE(held->published);
  CHECK(f.input->inspect().filling_batches == 1);
  held.reset();
  f.input->finish();
  f.input->finish();
  auto empty = f.input->claim();
  REQUIRE(empty);
  CHECK(empty->rows == 0);
  REQUIRE(f.input->exhausted());
}
TEST_CASE("native varlena validation ignores NULL garbage and checks live ranges", "[native_input]")
{
  input_fixture f({61, 0, 0, 1}, 128);
  auto batch = f.input->acquire(64, clock::now());
  uint8_t bytes[64]{};
  std::memset(bytes, 0xff, 24);
  bytes[24] = 3;
  std::memcpy(bytes + 25, "abc", 3);
  bytes[48] = 1;
  batch->write(0, {reinterpret_cast<const std::byte*>(bytes), 64});
  sirius_input_vector c{};
  c.data_bytes  = 48;
  c.null_offset = 48;
  c.null_bytes  = 8;
  f.input->publish(batch, 2, {&c, 1});
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->chars[0] == 3);
  CHECK(batch->is_null(0, 0));
  CHECK_FALSE(batch->is_null(0, 1));
  batch.reset();
  unit.reset();
  auto invalid = f.input->acquire(24, clock::now());
  uint32_t external[6]{UINT32_MAX, 10, 20};
  invalid->write(0, {reinterpret_cast<const std::byte*>(external), 24});
  c            = {};
  c.data_bytes = 24;
  REQUIRE_THROWS_AS(f.input->publish(invalid, 1, {&c, 1}), failure);
}
TEST_CASE("native publication and source release wake subscribers", "[native_input]")
{
  input_fixture f;
  auto wake  = std::make_shared<waker>();
  auto batch = f.input->acquire(8, clock::now());
  sirius_input_vector c{};
  c.data_bytes = 8;
  f.input->publish(batch, 1, {&c, 1});
  batch.reset();
  f.input->subscribe(wake);
  CHECK(wake->calls == 1);
  auto unit = f.input->claim();
  REQUIRE(unit);
  unit.reset();
  CHECK(wake->calls == 2);
  f.input->finish();
  CHECK(wake->calls == 3);
}
TEST_CASE("native constant vectors split lazily under expanded budget", "[native_input]")
{
  input_fixture f;
  auto batch = f.input->acquire(8, clock::now());
  sirius_input_vector c{};
  c.vector_class = SIRIUS_VECTOR_CONSTANT;
  c.data_bytes   = 8;
  f.input->publish(batch, 10000000, {&c, 1});
  batch.reset();
  auto first = f.input->claim();
  REQUIRE(first);
  CHECK(first->bytes <= input_target);
  CHECK(first->rows < 10000000);
  auto second = f.input->claim();
  REQUIRE(second);
  auto third = f.input->claim();  // may claim only the remaining few bytes
  CHECK(f.input->inspect().source_units <= 3);
  CHECK(f.input->inspect().retained_bytes == 8 + sizeof(sirius_input_vector));
}
TEST_CASE("live source wakeups are durable deduplicated and retire before plan destruction",
          "[native_input]")
{
  std::atomic<int> queued{0}, failed{0};
  source_wakeup wake([&] { ++queued; }, [&](std::exception_ptr) { ++failed; });
  wake.wake();
  std::jthread producer([&] {
    for (int i = 0; i < 10000; ++i)
      wake.wake();
  });
  producer.join();
  CHECK(queued == 1);
  wake.complete();
  CHECK(queued == 2);
  wake.complete();
  CHECK(queued == 2);
  wake.wake();
  CHECK(queued == 3);
  wake.close();
  wake.complete();
  wake.wake();
  CHECK(queued == 3);
  CHECK(failed == 0);
}
TEST_CASE("live wake dispatch failure seals subscription and reports failure once",
          "[native_input]")
{
  int failed = 0;
  source_wakeup wake([] { throw std::bad_alloc(); }, [&](std::exception_ptr) { ++failed; });
  wake.wake();
  wake.wake();
  wake.complete();
  CHECK(failed == 1);
}
TEST_CASE("native producer failure is not EOS and does not free filling bytes", "[native_input]")
{
  input_fixture f;
  auto batch = f.input->acquire(64, clock::now());
  f.input->stop(SIRIUS_EXECUTION_FAILED, "reader failed");
  f.input->discard();
  CHECK_FALSE(f.input->exhausted());
  CHECK(f.input->outcome().code == SIRIUS_EXECUTION_FAILED);
  CHECK(f.input->inspect().retained_bytes == 64 + sizeof(sirius_input_vector));
  REQUIRE_THROWS_AS(f.input->claim(), failure);
  batch.reset();
  CHECK(f.input->inspect().retained_bytes == 0);
}
TEST_CASE("native input allocation failures roll back all credit", "[native_input]")
{
  struct failing_pool : heap_pool {
    bool fail{true};
    std::unique_ptr<input_storage> allocate(std::size_t n) override
    {
      if (fail) throw std::bad_alloc();
      return heap_pool::allocate(n);
    }
  };
  auto pool  = std::make_shared<failing_pool>();
  auto input = std::make_shared<native_input>(1,
                                              std::vector<sirius_input_column>{{23, 0, 0, 1}},
                                              std::stop_token{},
                                              clock::now() + 10s,
                                              128,
                                              4);
  input->activate(pool);
  REQUIRE_THROWS_AS(input->acquire(64, clock::now()), std::bad_alloc);
  CHECK(input->inspect().retained_bytes == 0);
  CHECK(input->inspect().filling_batches == 0);
  pool->fail = false;
  auto lease = input->acquire(64, clock::now());
  CHECK(input->inspect().leases == 1);
  lease.reset();
  input->stop();
  input->discard();
}
TEST_CASE("native boundary distinguishes unprovable GPU quiescence from execution failure",
          "[native_input]")
{
  try {
    throw sirius::pipeline::gpu_stream_quiescence_error("unsafe native upload");
  } catch (...) {
    CHECK(current_error().code == SIRIUS_GPU_UNAVAILABLE);
  }
}
