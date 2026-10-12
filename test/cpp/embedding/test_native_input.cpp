/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/input.hpp"
#include "embedding/source_wakeup.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <catch.hpp>

#include <array>
#include <cstring>
#include <future>
#include <latch>
#include <limits>
#include <semaphore>
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
  std::vector<std::jthread> producers;
  explicit input_fixture(sirius_input_column column       = {23, 0, 0, 1},
                         std::size_t capacity             = 128 + sizeof(input_column_summary),
                         std::size_t target               = input_target,
                         std::size_t count                = 4,
                         std::shared_ptr<input_pool> pool = std::make_shared<heap_pool>())
    : input(std::make_shared<native_input>(1,
                                           std::vector{column},
                                           stop.get_token(),
                                           clock::now() + 10s,
                                           capacity,
                                           count,
                                           nullptr,
                                           target))
  {
    input->activate(std::move(pool));
  }
  ~input_fixture()
  {
    input->stop();
    input->discard();
    producers.clear();  // cancellation precedes joining, including failed assertions
  }
};
struct waker : capacity_waker {
  std::atomic<int> calls{0};
  void wake() noexcept override { ++calls; }
};
struct event_waker : capacity_waker {
  std::counting_semaphore<1024> events{0};
  void wake() noexcept override { events.release(); }
  bool next() { return events.try_acquire_for(2s); }
};
struct rounded_pool : heap_pool {
  std::size_t block;
  explicit rounded_pool(std::size_t size = 128) : block(size) {}
  std::size_t rounded(std::size_t bytes) const override
  {
    return (bytes + block - 1) / block * block;
  }
};
struct observed_storage : heap_storage {
  std::shared_ptr<std::atomic<int>> reads;
  std::shared_ptr<std::atomic<bool>> fail;
  observed_storage(std::size_t bytes,
                   std::shared_ptr<std::atomic<int>> count,
                   std::shared_ptr<std::atomic<bool>> failure)
    : heap_storage(bytes), reads(std::move(count)), fail(std::move(failure))
  {
  }
  void read(std::size_t offset, std::span<std::byte> bytes) const override
  {
    ++*reads;
    if (fail->load()) throw std::bad_alloc();
    input_storage::read(offset, bytes);
  }
};
struct observed_pool : heap_pool {
  std::shared_ptr<std::atomic<int>> reads = std::make_shared<std::atomic<int>>(0);
  std::shared_ptr<std::atomic<bool>> fail = std::make_shared<std::atomic<bool>>(false);
  std::unique_ptr<input_storage> allocate(std::size_t bytes) override
  {
    return std::make_unique<observed_storage>(bytes, reads, fail);
  }
};
constexpr std::size_t column_charge = sizeof(sirius_input_vector) + sizeof(input_column_summary);
void publish_ints(input_fixture& f, uint32_t rows, int64_t first = 0)
{
  auto batch = f.input->acquire(rows * sizeof(int64_t), clock::now());
  std::vector<int64_t> values(rows);
  for (uint32_t i = 0; i < rows; ++i)
    values[i] = first + i;
  batch->write(0, {reinterpret_cast<std::byte*>(values.data()), values.size() * sizeof(int64_t)});
  sirius_input_vector column{};
  column.data_bytes = values.size() * sizeof(int64_t);
  f.input->publish(batch, rows, {&column, 1});
}
}  // namespace
TEST_CASE("native input credits follow filling queued and retry owners", "[native_input]")
{
  input_fixture f({23, 0, 0, 1}, 144, 140);
  auto batch = f.input->acquire(64, clock::now());
  int64_t values[]{1, 2, 3, 4, 5, 6, 7, 8};
  batch->write(0, {reinterpret_cast<const std::byte*>(values), sizeof(values)});
  sirius_input_vector column{};
  column.data_bytes = 64;
  f.input->publish(batch, 8, {&column, 1});
  batch.reset();
  CHECK(f.input->inspect().retained_bytes == 64 + column_charge);
  CHECK(f.input->inspect().queued_batches == 1);
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 8);
  CHECK(f.input->inspect().queued_batches == 0);
  CHECK(f.input->inspect().retained_bytes == 64 + column_charge);
  REQUIRE_THROWS_AS(f.input->acquire(8, clock::now()), failure);
  auto retry = unit->slices[0].batch;
  unit.reset();
  CHECK(f.input->inspect().retained_bytes == 64 + column_charge);
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
  CHECK(f.input->inspect().retained_bytes == 64 + column_charge);
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
  input_fixture f({61, 0, 0, 1}, 144);
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
  REQUIRE_THROWS_AS(f.input->acquire(8, clock::now()), failure);  // pressure flushes the tail
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
  REQUIRE_THROWS_AS(f.input->acquire(64, clock::now()), failure);
  CHECK(wake->calls == 2);
  auto unit = f.input->claim();
  REQUIRE(unit);
  unit.reset();
  CHECK(wake->calls == 3);
  f.input->finish();
  CHECK(wake->calls == 4);
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
  CHECK_FALSE(f.input->claim());  // no tiny unit from residual expanded credit
  CHECK(f.input->inspect().source_units == 2);
  CHECK(f.input->inspect().retained_bytes == 8 + column_charge);
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
  CHECK(f.input->inspect().retained_bytes == 64 + column_charge);
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
                                              144,
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

TEST_CASE("native open inputs coalesce at the target and flush EOS tails",
          "[native_input][input_coalescing]")
{
  input_fixture f({23, 0, 0, 1}, 1024, 104, 128);
  for (int i = 0; i < 3; ++i) {
    publish_ints(f, 1, 100 + i);
    CHECK_FALSE(f.input->ready());
    CHECK_FALSE(f.input->claim());
  }
  publish_ints(f, 1, 103);
  REQUIRE(f.input->ready());
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 4);
  CHECK(unit->bytes == 104);
  REQUIRE(unit->slices.size() == 4);
  for (int i = 0; i < 4; ++i) {
    int64_t value{};
    unit->slices[i].batch->storage->read(0, {reinterpret_cast<std::byte*>(&value), sizeof(value)});
    CHECK(value == 100 + i);
  }
  unit.reset();
  CHECK(f.input->inspect().retained_bytes == 0);
  publish_ints(f, 1, 104);
  CHECK_FALSE(f.input->claim());
  f.input->finish();
  REQUIRE(f.input->ready());
  unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 1);
  unit.reset();
  CHECK(f.input->exhausted());
  CHECK(f.input->inspect().retained_bytes == 0);
}

TEST_CASE("native partial fronts preserve row order and remaining readiness",
          "[native_input][input_coalescing]")
{
  input_fixture f({23, 0, 0, 1}, 1024, 104);
  publish_ints(f, 10, 100);
  for (uint32_t begin : {0u, 4u}) {
    REQUIRE(f.input->ready());
    auto unit = f.input->claim();
    REQUIRE(unit);
    REQUIRE(unit->slices.size() == 1);
    CHECK(unit->slices[0].begin == begin);
    CHECK(unit->rows == 4);
    CHECK(unit->bytes == 104);
  }
  CHECK_FALSE(f.input->ready());
  CHECK_FALSE(f.input->claim());
  f.input->finish();
  auto tail = f.input->claim();
  REQUIRE(tail);
  CHECK(tail->rows == 2);
  CHECK(tail->slices[0].begin == 8);
  tail.reset();
  CHECK(f.input->inspect().retained_bytes == 0);
}

TEST_CASE("native zero-wait pressure flushes rounded credit and then retires",
          "[native_input][input_coalescing]")
{
  input_fixture f({23, 0, 0, 1}, 400, 512, 128, std::make_shared<rounded_pool>());
  publish_ints(f, 1);
  publish_ints(f, 1, 1);
  CHECK(f.input->inspect().retained_bytes == 400);
  CHECK_FALSE(f.input->ready());
  try {
    (void)f.input->acquire(8, clock::now());
    FAIL("full rounded credit must reject acquisition");
  } catch (failure const& e) {
    CHECK(e.error.code == SIRIUS_TIMEOUT);
  }
  REQUIRE(f.input->ready());
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 2);
  unit.reset();
  CHECK(f.input->inspect().retained_bytes == 0);
  publish_ints(f, 1, 2);
  CHECK_FALSE(f.input->ready());
  CHECK_FALSE(f.input->claim());
  f.input->finish();
  unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 1);
}

TEST_CASE("native successful small acquisition does not hide a larger waiter",
          "[native_input][input_coalescing]")
{
  std::promise<int> result;
  auto done = result.get_future();
  auto wake = std::make_shared<event_waker>();
  input_fixture f({23, 0, 0, 1}, 256, 512, 128);
  publish_ints(f, 1);
  f.input->subscribe(wake);
  REQUIRE(wake->next());
  f.producers.emplace_back([&] {
    try {
      {
        auto lease = f.input->acquire(128, clock::now() + 5s);
      }
      result.set_value(SIRIUS_OK);
    } catch (failure const& e) {
      result.set_value(e.error.code);
    }
  });
  REQUIRE(wake->next());  // registration is visible before the capacity wait
  auto smaller = f.input->acquire(8, clock::now());
  REQUIRE(f.input->ready());
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 1);
  unit.reset();
  CHECK(done.wait_for(0s) == std::future_status::timeout);
  smaller.reset();
  REQUIRE(done.wait_for(2s) == std::future_status::ready);
  CHECK(done.get() == SIRIUS_OK);
  f.producers.back().join();
  CHECK(f.input->inspect().retained_bytes == 0);
  publish_ints(f, 1);
  CHECK_FALSE(f.input->ready());
}

TEST_CASE("native pressure remains live until every capacity waiter retires",
          "[native_input][input_coalescing]")
{
  std::array<std::promise<std::shared_ptr<input_batch>>, 2> results;
  std::array done{results[0].get_future(), results[1].get_future()};
  std::counting_semaphore<2> granted{0};
  auto wake = std::make_shared<event_waker>();
  input_fixture f({23, 0, 0, 1}, 512, 1024, 128);
  publish_ints(f, 8);
  f.input->subscribe(wake);
  REQUIRE(wake->next());
  for (int i = 0; i < 2; ++i) {
    f.producers.emplace_back([&, i] {
      try {
        results[i].set_value(f.input->acquire(384, clock::now() + 5s));
      } catch (...) {
        results[i].set_exception(std::current_exception());
      }
      granted.release();
    });
    REQUIRE(wake->next());
  }
  auto smaller = f.input->acquire(8, clock::now());
  auto unit    = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 8);
  unit.reset();
  CHECK_FALSE(granted.try_acquire());  // the small filling lease still prevents either grant
  smaller.reset();
  REQUIRE(granted.try_acquire_for(2s));
  int first  = done[0].wait_for(0s) == std::future_status::ready ? 0 : 1;
  auto lease = done[first].get();
  sirius_input_vector column{};
  column.data_bytes = 8;
  f.input->publish(lease, 1, {&column, 1});
  REQUIRE(f.input->ready());  // grant cleared the latch, but the other waiter remains
  unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 1);
  lease.reset();
  unit.reset();
  REQUIRE(granted.try_acquire_for(2s));
  lease = done[1 - first].get();
  f.input->publish(lease, 1, {&column, 1});
  CHECK_FALSE(f.input->ready());
  CHECK_FALSE(f.input->claim());
  f.input->finish();
  unit = f.input->claim();
  REQUIRE(unit);
  lease.reset();
  unit.reset();
  for (auto& producer : f.producers)
    producer.join();
  CHECK(f.input->inspect().retained_bytes == 0);
}

TEST_CASE("native parked inputs report cancellation without flushing data",
          "[native_input][input_coalescing]")
{
  auto wake = std::make_shared<event_waker>();
  input_fixture f({23, 0, 0, 1}, 512, 1024);
  publish_ints(f, 1);
  f.input->subscribe(wake);
  REQUIRE(wake->next());
  CHECK_FALSE(f.input->ready());
  f.stop.request_stop();
  REQUIRE(wake->next());
  REQUIRE(f.input->ready());
  try {
    (void)f.input->claim();
    FAIL("cancelled input must preserve its terminal error");
  } catch (failure const& e) {
    CHECK(e.error.code == SIRIUS_CANCELLED);
  }
  f.input->discard();
  CHECK(f.input->inspect().retained_bytes == 0);
}

TEST_CASE("native whole-frame plans reuse validated string and NULL summaries",
          "[native_input][input_coalescing]")
{
  auto pool = std::make_shared<observed_pool>();
  input_fixture f({61, 0, 0, 1}, 512, 81, 4, pool);
  auto batch = f.input->acquire(64, clock::now());
  uint8_t bytes[64]{};
  std::memset(bytes, 0xff, 24);  // garbage under NULL must never be interpreted
  bytes[24] = 3;
  std::memcpy(bytes + 25, "abc", 3);
  bytes[48] = 1;
  batch->write(0, {reinterpret_cast<std::byte*>(bytes), sizeof(bytes)});
  sirius_input_vector column{};
  column.data_bytes  = 48;
  column.null_offset = 48;
  column.null_bytes  = 8;
  f.input->publish(batch, 2, {&column, 1});
  pool->reads->store(0);
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 2);
  CHECK(unit->bytes == 81);
  CHECK(unit->chars[0] == 3);
  CHECK(unit->nulls[0] == 1);
  CHECK(pool->reads->load() == 0);
}

TEST_CASE("native constant string summaries count logical rows and NULLs",
          "[native_input][input_coalescing]")
{
  for (uint32_t cls : {SIRIUS_VECTOR_CONSTANT, SIRIUS_VECTOR_NULL}) {
    input_fixture f({61, 0, 0, 1}, 256);
    auto batch = f.input->acquire(24, clock::now());
    std::byte bytes[24]{};
    bytes[0] = std::byte{2};
    bytes[1] = std::byte{'a'};
    bytes[2] = std::byte{'b'};
    if (cls == SIRIUS_VECTOR_NULL) std::memset(bytes, 0xff, sizeof(bytes));
    batch->write(0, bytes);
    sirius_input_vector column{};
    column.vector_class = cls;
    column.data_bytes   = 24;
    f.input->publish(batch, 4, {&column, 1});
    f.input->finish();
    auto unit = f.input->claim();
    REQUIRE(unit);
    CHECK(unit->rows == 4);
    CHECK(unit->chars[0] == (cls == SIRIUS_VECTOR_NULL ? 0 : 8));
    CHECK(unit->nulls[0] == (cls == SIRIUS_VECTOR_NULL ? 4 : 0));
  }
}

TEST_CASE("native descriptor fan-in flushes without requiring another producer acquire",
          "[native_input][input_coalescing]")
{
  input_fixture f({23, 0, 0, 1}, 16384, input_target, 128);
  for (int i = 0; i < 127; ++i)
    publish_ints(f, 1, i);
  CHECK_FALSE(f.input->ready());
  publish_ints(f, 1, 127);
  REQUIRE(f.input->ready());
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 128);
  CHECK(unit->slices.size() == 128);
  unit.reset();
  CHECK(f.input->inspect().retained_bytes == 0);
}

TEST_CASE("native acquisition timeout leaves a durable partial flush request",
          "[native_input][input_coalescing]")
{
  input_fixture f({23, 0, 0, 1}, 144, 512);
  publish_ints(f, 1);
  try {
    (void)f.input->acquire(64, clock::now() + 1s);
    FAIL("unreleased queued credit must time out");
  } catch (failure const& e) {
    CHECK(e.error.code == SIRIUS_TIMEOUT);
  }
  CHECK(f.input->inspect().blocked_acquires == 1);
  REQUIRE(f.input->ready());
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->rows == 1);
  unit.reset();
  publish_ints(f, 8);
  CHECK_FALSE(f.input->ready());  // neither waiter nor pressure latch survived
}

TEST_CASE("native failed split planning leaves its queued prefix and credit intact",
          "[native_input][input_coalescing]")
{
  auto pool = std::make_shared<observed_pool>();
  input_fixture f({23, 0, 0, 1}, 512, 104, 4, pool);
  auto batch = f.input->acquire(88, clock::now());
  sirius_input_vector column{};
  column.data_bytes  = 80;
  column.null_offset = 80;
  column.null_bytes  = 8;
  f.input->publish(batch, 10, {&column, 1});
  batch.reset();
  auto retained = f.input->inspect().retained_bytes;
  pool->fail->store(true);
  REQUIRE_THROWS_AS(f.input->claim(), std::bad_alloc);
  CHECK(f.input->inspect().retained_bytes == retained);
  CHECK(f.input->inspect().queued_batches == 1);
  CHECK(f.input->inspect().source_units == 0);
  pool->fail->store(false);
  auto unit = f.input->claim();
  REQUIRE(unit);
  CHECK(unit->slices[0].begin == 0);
  CHECK(unit->rows == 4);
  CHECK(unit->nulls[0] == 0);
}

TEST_CASE("native first rows retain soft-target and hard-bound behavior with wide summaries",
          "[native_input][input_coalescing]")
{
  for (std::size_t columns : {9u, 17u, 1024u}) {
    INFO("columns=" << columns);
    const uint32_t length = columns == 1024 ? 16u << 20 : 4u << 20;
    auto input =
      std::make_shared<native_input>(1,
                                     std::vector<sirius_input_column>(columns, {61, 0, 0, 1}),
                                     std::stop_token{},
                                     clock::now() + 5s);
    struct retire_input {
      std::shared_ptr<native_input> input;
      ~retire_input()
      {
        input->stop();
        input->discard();
      }
    } retire{input};
    input->activate(std::make_shared<heap_pool>());
    auto batch = input->acquire(length + 24, clock::now());
    uint32_t descriptor[6]{UINT32_MAX, 0, length};
    batch->write(0, {reinterpret_cast<std::byte*>(descriptor), sizeof(descriptor)});
    sirius_input_vector column{};
    column.vector_class = SIRIUS_VECTOR_CONSTANT;
    column.data_bytes   = 24;
    column.area_offset  = 24;
    column.area_bytes   = length;
    input->publish(
      batch, columns == 1024 ? INT32_MAX : 2, std::vector<sirius_input_vector>(columns, column));
    if (columns == 1024) {
      bool wide = batch->expanded_bytes > std::numeric_limits<std::size_t>::max();
      CHECK(wide);
    }
    batch.reset();
    REQUIRE(input->ready());
    if (columns == 9) {
      auto first = input->claim();
      REQUIRE(first);
      CHECK(first->rows == 1);
      CHECK(first->bytes > input_target);
      CHECK(first->bytes <= input_window);
      CHECK_FALSE(input->claim());
      CHECK(input->inspect().queued_batches == 1);
      first.reset();
      auto second = input->claim();
      REQUIRE(second);
      CHECK(second->slices[0].begin == 1);
      CHECK(second->rows == 1);
      second.reset();
      input->finish();
      CHECK(input->exhausted());
      CHECK(input->inspect().retained_bytes == 0);
    } else {
      try {
        (void)input->claim();
        FAIL("a single expanded row beyond the hard bound must fail");
      } catch (failure const& e) {
        CHECK(e.error.code == SIRIUS_RESOURCE_EXHAUSTED);
      }
      CHECK(input->inspect().queued_batches == 1);
      CHECK(input->inspect().source_units == 0);
    }
  }
}

TEST_CASE("native pool rounding beyond the window preserves resource exhaustion",
          "[native_input][input_coalescing]")
{
  input_fixture f({23, 0, 0, 1}, 160, input_target, 4, std::make_shared<rounded_pool>(256));
  try {
    (void)f.input->acquire(8, clock::now());
    FAIL("rounded physical storage cannot exceed the native window");
  } catch (failure const& e) {
    CHECK(e.error.code == SIRIUS_RESOURCE_EXHAUSTED);
  }
  CHECK(f.input->inspect().retained_bytes == 0);
  CHECK(f.input->inspect().filling_batches == 0);
}
