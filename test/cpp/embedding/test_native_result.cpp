/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/result.hpp"

#include <catch.hpp>

#include <cstring>
#include <future>
#include <latch>
using namespace sirius::embedding;
using namespace std::chrono_literals;
namespace {
struct result_test_storage final : input_storage {
  std::vector<std::byte> data;
  explicit result_test_storage(std::size_t n) : data(n) {}
  std::size_t size() const override { return data.size(); }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& fn) override
  {
    auto middle = data.size() / 2;
    fn(0, std::span(data).first(middle));
    fn(middle, std::span(data).subspan(middle));
  }
};
struct result_test_pool final : input_pool {
  std::size_t rounded(std::size_t n) const override { return (n + 7) / 8 * 8; }
  std::unique_ptr<input_storage> allocate(std::size_t n) override
  {
    return std::make_unique<result_test_storage>(rounded(n));
  }
};
struct result_fixture {
  std::shared_ptr<native_result> result = std::make_shared<native_result>();
  result_fixture() { result->activate(std::make_shared<result_test_pool>()); }
  ~result_fixture() { result->cancel(); }
};
}  // namespace
TEST_CASE("native result credit spans filling queued borrowed and cancellation", "[native_result]")
{
  result_fixture f;
  std::shared_ptr<result_batch> batch;
  REQUIRE(f.result->try_allocate(128, 1, batch) == SIRIUS_OK);
  auto used = f.result->inspect().retained_bytes;
  CHECK(used >= 128 + sizeof(sirius_input_vector));
  CHECK(f.result->inspect().filling_batches == 1);
  batch->rows = 2;
  std::array<std::byte, 128> data{};
  data[63] = std::byte{42};
  data[64] = std::byte{43};
  batch->storage->write(0, data);
  f.result->publish(std::move(batch));
  CHECK(f.result->inspect().queued_batches == 1);
  batch = f.result->next(clock::now());
  CHECK(f.result->inspect().borrowed_batches == 1);
  CHECK(f.result->inspect().retained_bytes == used);
  f.result->cancel();
  try {
    f.result->allocation_charge(8, 1);
    FAIL("cancelled publication remained active");
  } catch (failure const& error) {
    CHECK(error.error.code == SIRIUS_CANCELLED);
  }
  std::array<std::byte, 2> boundary{};
  batch->storage->read(63, boundary);
  CHECK(boundary[0] == std::byte{42});
  CHECK(boundary[1] == std::byte{43});
  CHECK(f.result->inspect().retained_bytes == used);
  batch.reset();
  CHECK(f.result->inspect().retained_bytes == 0);
  CHECK_FALSE(f.result->borrowed());
}
TEST_CASE("native result full window never lends dequeue credit", "[native_result]")
{
  result_fixture f;
  std::shared_ptr<result_batch> batch;
  REQUIRE(f.result->try_allocate(result_window - sizeof(result_batch), 0, batch) == SIRIUS_OK);
  f.result->publish(std::move(batch));
  auto held = f.result->next(clock::now());
  CHECK(f.result->try_allocate(1, 0, batch) == SIRIUS_TIMEOUT);
  CHECK_FALSE(batch);
  CHECK(f.result->inspect().peak_bytes <= result_window);
  held.reset();
  CHECK(f.result->try_allocate(1, 0, batch) == SIRIUS_OK);
}
TEST_CASE("native result zero rows are data and EOF follows quiescence", "[native_result]")
{
  result_fixture f;
  std::shared_ptr<result_batch> batch;
  REQUIRE(f.result->try_allocate(0, 0, batch) == SIRIUS_OK);
  f.result->publish(std::move(batch));
  CHECK(f.result->next(clock::now())->rows == 0);
  try {
    f.result->next(clock::now());
    FAIL("premature EOF");
  } catch (failure const& error) {
    CHECK(error.error.code == SIRIUS_TIMEOUT);
  }
  f.result->complete({});
  try {
    f.result->next(clock::now());
    FAIL("missing EOF");
  } catch (failure const& error) {
    CHECK(error.error.code == SIRIUS_EOF);
  }
}
TEST_CASE("native result errors discard queued data and wake pending pull", "[native_result]")
{
  result_fixture f;
  auto waiter = std::async(std::launch::async, [&] {
    try {
      f.result->next(clock::now() + 5s);
      return sirius_status(SIRIUS_OK);
    } catch (failure const& e) {
      return e.error.code;
    }
  });
  f.result->cancel();
  CHECK(waiter.get() == SIRIUS_CANCELLED);
  sirius_error fatal{};
  assign_error(fatal, SIRIUS_GPU_UNAVAILABLE, "fatal drain");
  f.result->complete(fatal);
  try {
    f.result->next(clock::now());
    FAIL("fatal outcome hidden");
  } catch (failure const& e) {
    CHECK(e.error.code == SIRIUS_GPU_UNAVAILABLE);
  }
}

namespace {
struct retirement_state {
  bool queue_batch{}, block_storage{}, block_pool{};
  std::latch entered{1}, resume{1};
  std::atomic<bool> pool_destroyed{false}, backend_destroyed{false}, pool_after_backend{false};
};
struct retirement_storage final : input_storage {
  std::shared_ptr<retirement_state> state;
  std::array<std::byte, 8> bytes{};
  explicit retirement_storage(std::shared_ptr<retirement_state> state) : state(std::move(state)) {}
  ~retirement_storage() override
  {
    if (state->block_storage) {
      state->entered.count_down();
      state->resume.wait();
    }
  }
  std::size_t size() const override { return bytes.size(); }
  void visit(std::function<void(std::size_t, std::span<std::byte>)> const& fn) override
  {
    fn(0, bytes);
  }
};
struct retirement_pool final : input_pool {
  std::shared_ptr<retirement_state> state;
  explicit retirement_pool(std::shared_ptr<retirement_state> state) : state(std::move(state)) {}
  ~retirement_pool() override
  {
    if (state->block_pool) {
      state->entered.count_down();
      state->resume.wait();
    }
    state->pool_after_backend = state->backend_destroyed.load();
    state->pool_destroyed     = true;
  }
  std::size_t rounded(std::size_t) const override { return 8; }
  std::unique_ptr<input_storage> allocate(std::size_t) override
  {
    return std::make_unique<retirement_storage>(state);
  }
};
struct retirement_driver final : query_driver {
  std::shared_ptr<native_result> result;
  bool publish;
  retirement_driver(std::shared_ptr<native_result> result, bool publish)
    : result(std::move(result)), publish(publish)
  {
  }
  void run(std::stop_token, clock::time_point) override
  {
    if (!publish) return;
    std::shared_ptr<result_batch> batch;
    auto status = result->try_allocate(8, 0, batch);
    if (status != SIRIUS_OK) throw failure(status, "test result allocation failed");
    batch->rows = 1;
    result->publish(std::move(batch));
  }
  void finish() override {}
};
struct retirement_backend final : engine_backend {
  std::shared_ptr<retirement_state> state;
  explicit retirement_backend(std::shared_ptr<retirement_state> state) : state(std::move(state)) {}
  ~retirement_backend() override { state->backend_destroyed = true; }
  std::unique_ptr<query_driver> prepare(std::string_view,
                                        std::stop_token,
                                        clock::time_point) override
  {
    throw failure(SIRIUS_INVALID_STATE, "test requires bound preparation");
  }
  std::unique_ptr<query_driver> prepare_bound(std::string_view,
                                              query_state const& query,
                                              input_registry&,
                                              std::stop_token,
                                              clock::time_point) override
  {
    query.results->activate(std::make_shared<retirement_pool>(state));
    return std::make_unique<retirement_driver>(query.results, state->queue_batch);
  }
};
struct retirement_fixture {
  std::shared_ptr<retirement_state> state = std::make_shared<retirement_state>();
  engine_control control{[this] { return std::make_unique<retirement_backend>(state); }, 1};
  std::shared_ptr<query_state> query;
  retirement_fixture()
  {
    if (control.initialize().code != SIRIUS_OK)
      throw std::runtime_error("test initialization failed");
    query = control.create("retirement", 10s);
  }
  ~retirement_fixture()
  {
    control.close_query(query, 5s);
    control.close(5s);
  }
  void run()
  {
    REQUIRE(control.prepare(query, 5s).code == SIRIUS_OK);
    REQUIRE(control.start(query).code == SIRIUS_OK);
    REQUIRE(control.wait(query, 5s).code == SIRIUS_OK);
  }
};
struct resume_retirement {
  std::latch& latch;
  bool released{};
  void release()
  {
    if (!released) {
      released = true;
      latch.count_down();
    }
  }
  ~resume_retirement() { release(); }
};
}  // namespace

TEST_CASE("completed result facade releases its pool before backend retirement", "[native_result]")
{
  retirement_fixture f;
  f.run();
  REQUIRE(f.state->pool_destroyed.load());
  REQUIRE_FALSE(f.state->backend_destroyed.load());
  REQUIRE(f.control.close_query(f.query, 5s).code == SIRIUS_OK);
  REQUIRE(f.query->inputs != nullptr);
  REQUIRE(f.query->results != nullptr);
  REQUIRE(f.control.close(5s).code == SIRIUS_OK);
  REQUIRE(f.state->backend_destroyed.load());
  CHECK_FALSE(f.state->pool_after_backend.load());
  // The still-live query/facade shared pointers now own only CPU state.
  CHECK(f.query->results->inspect().leases == 0);
}

TEST_CASE("queued result discard keeps query busy until physical retirement", "[native_result]")
{
  retirement_fixture f;
  f.state->queue_batch   = true;
  f.state->block_storage = true;
  f.run();
  auto cancellation = std::async(std::launch::async, [&] { f.control.cancel(f.query); });
  resume_retirement resume{f.state->resume};
  f.state->entered.wait();
  auto state = f.query->results->inspect();
  CHECK(state.borrowed_batches == 0);
  CHECK(state.queued_batches == 0);
  CHECK(state.leases == 1);
  CHECK(f.control.close_query(f.query, 0ms).code == SIRIUS_BUSY);
  CHECK(f.control.close(0ms).code == SIRIUS_BUSY);
  CHECK_FALSE(f.state->backend_destroyed.load());
  resume.release();
  cancellation.get();
  CHECK(f.state->pool_destroyed.load());
  CHECK(f.query->results->inspect().leases == 0);
  REQUIRE(f.control.close_query(f.query, 5s).code == SIRIUS_OK);
  REQUIRE(f.control.close(5s).code == SIRIUS_OK);
  CHECK_FALSE(f.state->pool_after_backend.load());
}

TEST_CASE("borrowed result remains counted while its pool destructor runs", "[native_result]")
{
  retirement_fixture f;
  f.state->queue_batch = true;
  f.state->block_pool  = true;
  f.run();
  auto batch = f.query->results->next(clock::now());
  auto releasing =
    std::async(std::launch::async, [batch = std::move(batch)]() mutable { batch.reset(); });
  resume_retirement resume{f.state->resume};
  f.state->entered.wait();
  auto state = f.query->results->inspect();
  CHECK(state.borrowed_batches == 1);
  CHECK(state.leases == 1);
  CHECK(f.control.close_query(f.query, 0ms).code == SIRIUS_BUSY);
  CHECK_FALSE(f.state->backend_destroyed.load());
  resume.release();
  releasing.get();
  REQUIRE(f.control.close_query(f.query, 5s).code == SIRIUS_OK);
  REQUIRE(f.control.close(5s).code == SIRIUS_OK);
  CHECK_FALSE(f.state->pool_after_backend.load());
}
