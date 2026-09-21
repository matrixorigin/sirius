/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding/buffer_budget.hpp"
#include "embedding/control.hpp"
#include "pipeline/completion_handler.hpp"

#include <catch.hpp>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <future>
#include <latch>
#include <limits>
#include <vector>

using namespace std::chrono_literals;
using namespace sirius::embedding;

TEST_CASE("native fatal status survives an earlier completion signal",
          "[native_control][completion_handler]")
{
  sirius::pipeline::completion_handler handler;
  auto future = handler.get_awaitable();
  handler.mark_completed();
  auto fatal = std::make_exception_ptr(std::runtime_error("fatal synchronization"));
  handler.report_fatal_error(fatal);
  REQUIRE_NOTHROW(future.get());
  REQUIRE(handler.has_error());
  REQUIRE(handler.fatal_error() == fatal);
  handler.report_fatal_error(std::make_exception_ptr(std::runtime_error("later")));
  REQUIRE(handler.fatal_error() == fatal);
}

namespace {
struct recording {
  std::thread::id thread;
  std::atomic<int> prepared{0}, ran{0}, finished{0}, destroyed{0};
  std::atomic<bool> wrong_thread{false};
  std::latch running{1};
  bool block{false};
  bool fail_run{false};
  bool fail_finish{false};
  bool startable{true};
  bool runtime_available{true};
};
class test_driver final : public query_driver {
 public:
  explicit test_driver(std::shared_ptr<recording> r) : r_(std::move(r)) {}
  ~test_driver() override
  {
    check_thread();
    ++r_->destroyed;
  }
  void run(std::stop_token stop, clock::time_point deadline) override
  {
    check_thread();
    ++r_->ran;
    r_->running.count_down();
    if (r_->block) {
      std::mutex mutex;
      std::condition_variable_any changed;
      std::unique_lock lock(mutex);
      changed.wait_until(lock, stop, deadline, [] { return false; });
    }
    if (r_->fail_run) throw std::runtime_error("injected execution error");
  }
  void finish() override
  {
    check_thread();
    if (r_->fail_finish) throw std::runtime_error("injected unprovable cleanup");
    ++r_->finished;
  }
  bool startable() const noexcept override { return r_->startable; }

 private:
  void check_thread()
  {
    if (std::this_thread::get_id() != r_->thread) r_->wrong_thread = true;
  }
  std::shared_ptr<recording> r_;
};
class test_backend final : public engine_backend {
 public:
  explicit test_backend(std::shared_ptr<recording> r) : r_(std::move(r))
  {
    r_->thread = std::this_thread::get_id();
  }
  std::unique_ptr<query_driver> prepare(std::string_view plan,
                                        std::stop_token,
                                        clock::time_point) override
  {
    ++r_->prepared;
    if (plan == "bad") throw failure(SIRIUS_INVALID_ARGUMENT, "invalid test plan");
    if (plan == "poisoned") {
      r_->runtime_available = false;
      throw failure(SIRIUS_UNSUPPORTED, "failure after execution-window health was poisoned");
    }
    return std::make_unique<test_driver>(r_);
  }
  bool available() const noexcept override { return r_->runtime_available; }

 private:
  std::shared_ptr<recording> r_;
};
struct fixture {
  std::shared_ptr<recording> record = std::make_shared<recording>();
  engine_control control{[this] { return std::make_unique<test_backend>(record); }, 2};
  std::vector<std::shared_ptr<query_state>> queries;
  fixture()
  {
    if (control.initialize().code != SIRIUS_OK) throw std::runtime_error("test init");
  }
  ~fixture()
  {
    for (auto const& q : queries)
      (void)control.close_query(q, 10s);
    (void)control.close(10s);
  }
  std::shared_ptr<query_state> create(std::string_view plan = "ok")
  {
    auto q = control.create(plan, 10s);
    queries.push_back(q);
    return q;
  }
};
}  // namespace

TEST_CASE("native coordinator keeps preparation execution and destruction on one thread",
          "[native_control]")
{
  fixture f;
  auto q = f.create();
  REQUIRE(f.control.prepare(q, 10s).code == SIRIUS_OK);
  auto start = std::async(std::launch::async, [&] { return f.control.start(q); });
  REQUIRE(start.get().code == SIRIUS_OK);
  REQUIRE(f.control.wait(q, 10s).code == SIRIUS_OK);
  CHECK(f.record->prepared == 1);
  CHECK(f.record->ran == 1);
  CHECK(f.record->finished == 1);
  CHECK(f.record->destroyed == 1);
  CHECK_FALSE(f.record->wrong_thread);
  CHECK(f.control.start(q).code == SIRIUS_INVALID_STATE);
}

TEST_CASE("prepared admission can remain explicitly non-startable", "[native_control]")
{
  fixture f;
  f.record->startable = false;
  auto q              = f.create();
  REQUIRE(f.control.prepare(q, 10s).code == SIRIUS_OK);
  REQUIRE(q->phase == query_phase::PREPARED);
  REQUIRE(f.control.start(q).code == SIRIUS_UNSUPPORTED);
  REQUIRE(q->phase == query_phase::PREPARED);
  f.control.cancel(q);
  REQUIRE(f.control.wait(q, 10s).code == SIRIUS_CANCELLED);
  CHECK(f.record->ran == 0);
  CHECK(f.record->finished == 1);
}

TEST_CASE("native contracts synchronously copy bounded descriptors", "[native_control]")
{
  fixture f;
  auto q             = f.create();
  char query_id[]    = "query";
  char output_name[] = "out";
  sirius_column output{23, 0, 0, 0, output_name, 3, 0};
  sirius_query_contract contract{
    sizeof(contract), SIRIUS_ABI_VERSION, 7, 0, query_id, 5, {0}, &output, 1};
  f.control.bind_query(q, contract);
  query_id[0] = output_name[0] = 'X';
  CHECK(q->contract->query_id == "query");
  CHECK(q->contract->outputs[0].name == "out");

  char column_name[] = "c";
  sirius_read_column column{{23, 0, 0, 0, column_name, 1, 0}, 99, 4, 0};
  sirius_read_binding read{sizeof(read),
                           SIRIUS_ABI_VERSION,
                           42,
                           SIRIUS_READ_MO,
                           0,
                           "db",
                           2,
                           "t",
                           1,
                           "s",
                           1,
                           &column,
                           1,
                           nullptr,
                           0,
                           nullptr,
                           0};
  f.control.register_read(q, read);
  column_name[0] = 'X';
  CHECK(q->bindings[0].columns[0].logical.name == "c");
  REQUIRE_THROWS_AS(f.control.register_read(q, read), failure);
}

TEST_CASE("native query identity is an owned opaque byte string", "[native_control]")
{
  fixture f;
  auto q = f.create();
  char query_id[]{'\0', 'q', '\0', 'i', '\0'};
  const std::string expected(query_id, sizeof(query_id));
  sirius_query_contract contract{
    sizeof(contract), SIRIUS_ABI_VERSION, 7, 0, query_id, sizeof(query_id), {0}, nullptr, 0};

  f.control.bind_query(q, contract);
  std::fill(std::begin(query_id), std::end(query_id), 'X');

  REQUIRE(q->contract->query_id.size() == expected.size());
  CHECK(q->contract->query_id == expected);
}

TEST_CASE("native query identity retains its nonempty bounded contract", "[native_control]")
{
  fixture f;
  const char query_id = 'q';
  sirius_query_contract contract{
    sizeof(contract), SIRIUS_ABI_VERSION, 7, 0, &query_id, 0, {0}, nullptr, 0};

  auto empty = f.create();
  REQUIRE_THROWS_AS(f.control.bind_query(empty, contract), failure);

  contract.query_id_bytes = 4097;
  auto oversized          = f.create();
  REQUIRE_THROWS_AS(f.control.bind_query(oversized, contract), failure);

  contract.query_id       = nullptr;
  contract.query_id_bytes = 1;
  auto missing            = f.create();
  REQUIRE_THROWS_AS(f.control.bind_query(missing, contract), failure);
}

TEST_CASE("native text fields still reject embedded NUL bytes", "[native_control]")
{
  fixture f;
  auto q              = f.create();
  const char query_id = 'q';
  const char output_name[]{'o', '\0', 'u'};
  sirius_column output{23, 0, 0, 0, output_name, sizeof(output_name), 0};
  sirius_query_contract contract{
    sizeof(contract), SIRIUS_ABI_VERSION, 7, 0, &query_id, 1, {0}, &output, 1};

  REQUIRE_THROWS_AS(f.control.bind_query(q, contract), failure);
}

TEST_CASE("native queued cancellation bypasses blocked active execution", "[native_control]")
{
  fixture f;
  f.record->block = true;
  auto active     = f.create();
  REQUIRE(f.control.prepare(active, 10s).code == SIRIUS_OK);
  REQUIRE(f.control.start(active).code == SIRIUS_OK);
  f.record->running.wait();
  auto queued = f.create();
  REQUIRE(f.control.prepare(queued, 0ms).code == SIRIUS_TIMEOUT);
  REQUIRE(f.control.inspect().queued_queries == 1);
  f.control.cancel(queued);
  REQUIRE(f.control.wait(queued, 0ms).code == SIRIUS_CANCELLED);
  CHECK(f.record->prepared == 1);
  CHECK(f.control.inspect().queued_queries == 0);
  CHECK(f.control.wait(active, 0ms).code == SIRIUS_TIMEOUT);
  f.control.cancel(active);
  REQUIRE(f.control.wait(active, 10s).code == SIRIUS_CANCELLED);
  CHECK(f.record->finished == 1);
}

TEST_CASE("native preparation and execution failures release ownership before reuse",
          "[native_control]")
{
  fixture f;
  auto bad = f.create("bad");
  REQUIRE(f.control.prepare(bad, 10s).code == SIRIUS_INVALID_ARGUMENT);
  REQUIRE(f.control.close_query(bad, 10s).code == SIRIUS_OK);
  f.record->fail_run = true;
  auto q             = f.create();
  REQUIRE(f.control.prepare(q, 10s).code == SIRIUS_OK);
  REQUIRE(f.control.start(q).code == SIRIUS_OK);
  REQUIRE(f.control.wait(q, 10s).code == SIRIUS_EXECUTION_FAILED);
  CHECK(f.record->finished == 1);
  CHECK(f.record->destroyed == 1);
  REQUIRE(f.control.close_query(q, 10s).code == SIRIUS_OK);
  auto unstarted = f.create();
  REQUIRE(f.control.prepare(unstarted, 10s).code == SIRIUS_OK);
  REQUIRE(f.control.close_query(unstarted, 10s).code == SIRIUS_OK);
  CHECK(f.record->finished == 2);
  CHECK(f.control.inspect().live_queries == 0);
}

TEST_CASE("native handle capacity and engine close fail closed", "[native_control]")
{
  fixture f;
  f.create();
  f.create();
  f.create();
  REQUIRE_THROWS_AS(f.create(), failure);
  CHECK(f.control.inspect().live_queries == 3);
  REQUIRE(f.control.close(0ms).code == SIRIUS_BUSY);
  REQUIRE_THROWS_AS(f.create(), failure);
  for (auto const& q : f.queries)
    REQUIRE(f.control.wait(q, 0ms).code == SIRIUS_CANCELLED);
}

TEST_CASE("native initialization exceptions are observed after worker exit", "[native_control]")
{
  engine_control control([]() -> std::unique_ptr<engine_backend> { throw std::bad_alloc(); }, 1);
  REQUIRE(control.initialize().code == SIRIUS_RESOURCE_EXHAUSTED);
}

TEST_CASE("expired native queries are releasable rather than confused with wait timeout",
          "[native_control]")
{
  fixture f;
  auto q = f.control.create("ok", 1ms);
  f.queries.push_back(q);
  (void)f.control.prepare(q, 10s);
  REQUIRE(f.control.wait(q, 10s).code == SIRIUS_TIMEOUT);
  REQUIRE(f.control.close_query(q, 10s).code == SIRIUS_OK);
  CHECK(f.control.inspect().live_queries == 0);
}

TEST_CASE("native execution returning at its deadline is not successful", "[native_control]")
{
  fixture f;
  f.record->block = true;
  auto q          = f.control.create("ok", 2s);
  f.queries.push_back(q);
  REQUIRE(f.control.prepare(q, 10s).code == SIRIUS_OK);
  REQUIRE(f.control.start(q).code == SIRIUS_OK);
  REQUIRE(f.control.wait(q, 10s).code == SIRIUS_TIMEOUT);
  CHECK(f.record->ran == 1);
  CHECK(f.record->finished == 1);
  CHECK(f.record->destroyed == 1);
  REQUIRE(f.control.close_query(q, 10s).code == SIRIUS_OK);
  CHECK(f.control.inspect().live_queries == 0);
}

TEST_CASE("unprovable native cleanup retains owners and seals admission",
          "[native_control][fatal_process]")
{
  // Fatal recovery deliberately retains native owners until process death.
  // Exercise that contract in a test-owned child; do not poison or leak the
  // parent test runtime, and bound the child's lifetime independently.
  auto pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    alarm(10);
    auto r         = std::make_shared<recording>();
    r->fail_finish = true;
    auto* control  = new engine_control([r] { return std::make_unique<test_backend>(r); }, 1);
    if (control->initialize().code != SIRIUS_OK) _exit(1);
    auto q = control->create("ok", 5s);
    if (control->prepare(q, 5s).code != SIRIUS_OK || control->start(q).code != SIRIUS_OK) _exit(2);
    if (control->wait(q, 5s).code != SIRIUS_GPU_UNAVAILABLE) _exit(3);
    if (q->phase != query_phase::UNAVAILABLE || r->destroyed != 0) _exit(4);
    auto execution = control->inspect_execution(q);
    if (!execution.terminal || !execution.fatal ||
        execution.terminal_status != SIRIUS_GPU_UNAVAILABLE)
      _exit(10);
    if (control->close_query(q, 0ms).code != SIRIUS_GPU_UNAVAILABLE) _exit(5);
    auto stats = control->inspect();
    if (!stats.unavailable || stats.accepting_queries || stats.live_queries != 1) _exit(6);
    if (control->close(0ms).code != SIRIUS_GPU_UNAVAILABLE) _exit(7);
    try {
      (void)control->create("ok", 5s);
      _exit(8);
    } catch (failure const& e) {
      if (e.error.code != SIRIUS_GPU_UNAVAILABLE) _exit(9);
    }
    _exit(0);
  }
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("poisoned preparation cannot masquerade as unsupported",
          "[native_control][fatal_process]")
{
  auto pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    alarm(10);
    auto record = std::make_shared<recording>();
    auto* control =
      new engine_control([record] { return std::make_unique<test_backend>(record); }, 1);
    if (control->initialize().code != SIRIUS_OK) _exit(1);
    auto query = control->create("poisoned", 5s);
    if (control->prepare(query, 5s).code != SIRIUS_GPU_UNAVAILABLE) _exit(2);
    auto execution = control->inspect_execution(query);
    if (!execution.terminal || !execution.fatal ||
        execution.terminal_status != SIRIUS_GPU_UNAVAILABLE)
      _exit(7);
    auto stats = control->inspect();
    if (!stats.unavailable || stats.accepting_queries) _exit(3);
    if (control->close_query(query, 0ms).code != SIRIUS_GPU_UNAVAILABLE) _exit(4);
    try {
      control->create("ok", 5s);
      _exit(5);
    } catch (failure const& e) {
      if (e.error.code != SIRIUS_GPU_UNAVAILABLE) _exit(6);
    }
    _exit(0);
  }
  int status{};
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("native buffer credit follows moved leases and outlives budget facade", "[native_credit]")
{
  buffer_budget::lease first, moved;
  {
    buffer_budget budget(8, 1);
    REQUIRE(budget.acquire(8, {}, clock::now(), first) == SIRIUS_OK);
    buffer_budget::lease rejected;
    REQUIRE(budget.acquire(1, {}, clock::now(), rejected) == SIRIUS_TIMEOUT);
    REQUIRE(budget.acquire(std::numeric_limits<std::size_t>::max(), {}, clock::now(), rejected) ==
            SIRIUS_RESOURCE_EXHAUSTED);
    moved = std::move(first);
    CHECK_FALSE(first);
    CHECK(budget.inspect().bytes == 8);
    CHECK(budget.inspect().leases == 1);
    budget.close();
    REQUIRE(budget.acquire(1, {}, clock::now(), rejected) == SIRIUS_CANCELLED);
  }
  moved.reset();
  CHECK_FALSE(moved);
}

TEST_CASE("native buffer cancellation wakes a full budget without releasing the held lease",
          "[native_credit]")
{
  buffer_budget budget(8, 1);
  buffer_budget::lease held;
  REQUIRE(budget.acquire(8, {}, clock::now(), held) == SIRIUS_OK);
  std::stop_source stop;
  auto waiting = std::async(std::launch::async, [&] {
    buffer_budget::lease extra;
    return budget.acquire(1, stop.get_token(), clock::now() + 10s, extra);
  });
  stop.request_stop();
  REQUIRE(waiting.get() == SIRIUS_CANCELLED);
  CHECK(budget.inspect().bytes == 8);
  held.reset();
  CHECK(budget.inspect().bytes == 0);
  CHECK(budget.inspect().leases == 0);
}
