/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/control.hpp"
#include "embedding/tae_demand.hpp"
#include "embedding/tae_read.hpp"

#include <catch.hpp>

#include <array>
#include <atomic>
#include <future>
#include <limits>
#include <vector>

using namespace sirius::embedding;
using namespace std::chrono_literals;

TEST_CASE("TAE permits bound all sources and survive dequeue retry and controller close",
          "[tae_demand][cpu]")
{
  auto budget     = std::make_shared<buffer_budget>(tae_host_capacity, 2);
  auto controller = make_tae_demand_controller(1, budget);
  std::promise<std::shared_ptr<tae_work_permit>> first, second, third;
  auto a = first.get_future();
  auto b = second.get_future();
  auto c = third.get_future();
  REQUIRE(controller->request([&](auto p) { first.set_value(std::move(p)); }));
  REQUIRE(controller->request([&](auto p) { second.set_value(std::move(p)); }));
  REQUIRE(a.wait_for(5s) == std::future_status::ready);
  REQUIRE(b.wait_for(5s) == std::future_status::ready);
  auto one = a.get();
  auto two = b.get();
  REQUIRE(one);
  REQUIRE(two);
  REQUIRE(controller->request([&](auto p) { third.set_value(std::move(p)); }));
  CHECK(c.wait_for(0s) == std::future_status::timeout);
  auto state = controller->inspect();
  CHECK(state.active == 2);
  CHECK(state.queued == 1);
  CHECK(state.peak == state.limit);
  CHECK(budget->inspect().bytes == 2 * tae_max_slice);

  auto retry         = one;
  bool storage_freed = false;
  one->retain_staging(std::shared_ptr<int>(new int(1), [&](int* value) {
    // Physical bytes go away before their entitlement is returned.
    storage_freed = true;
    CHECK(budget->inspect().leases == 2);
    delete value;
  }));
  one.reset();
  CHECK_FALSE(storage_freed);
  CHECK(c.wait_for(0s) == std::future_status::timeout);
  retry.reset();
  CHECK(storage_freed);
  REQUIRE(c.wait_for(5s) == std::future_status::ready);
  auto three = c.get();
  REQUIRE(three);
  CHECK(controller->inspect().peak == 2);
  controller->close();
  CHECK_FALSE(controller->request([](auto) {}));
  controller.reset();
  // The accounting generation outlives the controller facade.
  CHECK(budget->inspect().leases == 2);
  two.reset();
  three.reset();
  CHECK(budget->inspect().bytes == 0);
  CHECK(budget->inspect().leases == 0);
}

TEST_CASE("TAE controller close drops queued metadata without waiting for retained work",
          "[tae_demand][cpu]")
{
  auto budget     = std::make_shared<buffer_budget>(8192, 2);
  auto controller = make_tae_demand_controller(1, budget, 8192);
  std::promise<std::shared_ptr<tae_work_permit>> first, second;
  auto a = first.get_future();
  auto b = second.get_future();
  controller->request([&](auto p) { first.set_value(std::move(p)); });
  controller->request([&](auto p) { second.set_value(std::move(p)); });
  REQUIRE(a.wait_for(5s) == std::future_status::ready);
  REQUIRE(b.wait_for(5s) == std::future_status::ready);
  auto one = a.get();
  auto two = b.get();
  std::atomic<bool> later_scan_started{false};
  controller->request([&](auto) { later_scan_started = true; });
  controller->close();
  CHECK_FALSE(later_scan_started.load());
  CHECK(controller->inspect().queued == 0);
  CHECK(controller->inspect().active == 2);
}

TEST_CASE("TAE staging entitlements fit 64 MiB for every supported stream count",
          "[tae_demand][cpu]")
{
  for (auto streams : {0u, 1u, 2u, 8u, 64u, 128u}) {
    auto budget     = std::make_shared<buffer_budget>(tae_host_capacity, 256);
    auto controller = make_tae_demand_controller(streams, budget);
    auto state      = controller->inspect();
    CHECK(state.limit == std::max<std::size_t>(2, 2 * streams));
    CHECK(state.slice_bytes <= tae_max_slice);
    CHECK(state.slice_bytes >= 2048);
    CHECK(state.slice_bytes % 2048 == 0);
    CHECK(state.slice_bytes * state.limit <= tae_host_capacity);
  }
  auto budget = std::make_shared<buffer_budget>(4096, 2);
  CHECK_THROWS(make_tae_demand_controller(2, budget, 4096));
  CHECK_THROWS(make_tae_demand_controller(std::numeric_limits<std::size_t>::max(), budget));
}

TEST_CASE("TAE metadata cache evicts old immutable objects within its query bound",
          "[tae_demand][cpu]")
{
  auto budget     = std::make_shared<buffer_budget>(tae_host_capacity, 2);
  auto controller = make_tae_demand_controller(1, budget);
  auto first      = std::make_shared<const int>(42);
  controller->cache_metadata("first", first, 20u << 20);
  CHECK(controller->cached_metadata("first") == first);
  controller->cache_metadata("second", std::make_shared<const int>(43), 20u << 20);
  CHECK_FALSE(controller->cached_metadata("first"));
  CHECK(controller->inspect().cached_metadata_bytes ==
        (20u << 20) + 512 + 2 * std::string("second").size());
  CHECK(*first == 42);
  CHECK_THROWS(controller->cache_metadata("large", first, 33u << 20));
}

TEST_CASE("TAE runtime metadata is charged once within the existing engine budget",
          "[tae_demand][native_control][cpu]")
{
  struct metadata_backend final : engine_backend {
    std::unique_ptr<query_driver> prepare(std::string_view,
                                          std::stop_token,
                                          clock::time_point) override
    {
      throw std::runtime_error("metadata admission test does not prepare");
    }
  };
  engine_control control([] { return std::make_unique<metadata_backend>(); }, 2);
  REQUIRE(control.initialize().code == SIRIUS_OK);
  auto first  = control.create("plan", 10s);
  auto second = control.create("plan", 10s);
  sirius_read_binding binding{};
  binding.struct_size         = sizeof(binding);
  binding.abi_version         = SIRIUS_ABI_VERSION;
  binding.binding_id          = 1;
  binding.source_kind         = SIRIUS_READ_TAE;
  binding.database_name       = "d";
  binding.database_name_bytes = 1;
  binding.table_name          = "t";
  binding.table_name_bytes    = 1;
  binding.data_root           = "/data";
  binding.data_root_bytes     = 5;
  binding.tae_manifest        = "{}";
  binding.tae_manifest_bytes  = 2;
  auto before                 = first->metadata_bytes;
  control.register_read(first, binding);
  auto initial_charge = first->metadata_bytes - before;
  CHECK(initial_charge > tae_metadata_reservation_bytes);
  auto lease_count   = first->metadata_leases.size();
  binding.binding_id = 2;
  control.register_read(first, binding);
  CHECK(first->metadata_bytes - before - initial_charge ==
        initial_charge - tae_metadata_reservation_bytes);
  CHECK(first->metadata_leases.size() == lease_count + 1);
  auto const second_before = second->metadata_bytes;
  CHECK_THROWS_AS(control.register_read(second, binding), failure);
  CHECK(second->bindings.empty());
  CHECK(second->metadata_bytes == second_before);
  REQUIRE(control.close_query(first, 10s).code == SIRIUS_OK);
  CHECK(first->metadata_leases.empty());
  REQUIRE_NOTHROW(control.register_read(second, binding));
  REQUIRE(control.close_query(second, 10s).code == SIRIUS_OK);
  REQUIRE(control.close(10s).code == SIRIUS_OK);
}

TEST_CASE("TAE converter metadata entitlements scale down at 128 streams", "[tae_demand][cpu]")
{
  auto budget     = std::make_shared<buffer_budget>(tae_host_capacity, 256);
  auto controller = make_tae_demand_controller(128, budget);
  std::promise<std::shared_ptr<tae_work_permit>> promised;
  auto ready = promised.get_future();
  controller->request([&](auto permit) { promised.set_value(std::move(permit)); });
  REQUIRE(ready.wait_for(5s) == std::future_status::ready);
  auto permit = ready.get();
  REQUIRE(permit);
  CHECK(permit->metadata_bytes() == (256u << 10));
  CHECK(permit->metadata_bytes() * controller->inspect().limit == tae_work_metadata_total);
  CHECK(permit->chunk_limit(64) == 48);
  CHECK(permit->chunk_limit(64) * tae_chunk_metadata_charge + tae_work_fixed_metadata <=
        permit->metadata_bytes());
  CHECK(permit->chunk_limit(64) * 64 <= tae_chunk_vector_limit);
  static_assert(tae_metadata_reservation_bytes == (128u << 20));
}

TEST_CASE("TAE CRC slices strip in place across every boundary and partial last block",
          "[tae_slices][cpu]")
{
  std::vector<std::uint8_t> logical(2044 * 4 + 37);
  for (std::size_t i = 0; i < logical.size(); ++i)
    logical[i] = (i * 17) % 251;
  std::vector<std::uint8_t> physical;
  for (std::size_t i = 0; i < logical.size(); i += 2044) {
    physical.insert(physical.end(), 4, 0xa5);
    auto n = std::min<std::size_t>(2044, logical.size() - i);
    physical.insert(physical.end(), logical.begin() + i, logical.begin() + i + n);
  }
  std::size_t largest_read = 0;
  tae_read_at reader       = [&](std::uint64_t offset, std::size_t count, std::uint8_t* target) {
    largest_read = std::max(largest_read, count);
    if (offset > physical.size() || count > physical.size() - offset) return std::size_t{0};
    std::memcpy(target, physical.data() + offset, count);
    return count;
  };
  for (auto begin : {0u, 1u, 2043u, 2044u, 2045u, 8175u}) {
    std::array<std::uint8_t, 4096> scratch;
    std::vector<std::uint8_t> restored;
    auto offset = std::size_t(begin);
    while (offset < logical.size()) {
      auto n =
        read_tae_slice(reader, physical.size(), true, offset, logical.size() - offset, scratch);
      REQUIRE(n > 0);
      restored.insert(restored.end(), scratch.begin(), scratch.begin() + n);
      offset += n;
    }
    CHECK(restored == std::vector<std::uint8_t>(logical.begin() + begin, logical.end()));
  }
  CHECK(largest_read <= 4096);
  tae_read_at metadata_reader = [&](std::uint64_t offset, std::size_t count, std::uint8_t* target) {
    // A 64-byte logical header ends BEFORE the first payload. Do not issue a
    // whole physical CRC-block read just because they share that block.
    CHECK(offset == 4);
    CHECK(count == 64);
    std::memcpy(target, physical.data() + offset, count);
    return count;
  };
  std::array<std::uint8_t, 2048> scratch;
  CHECK(read_tae_slice(metadata_reader, physical.size(), true, 0, 64, scratch) == 64);
  CHECK_THROWS(read_tae_slice(reader, physical.size(), true, logical.size() - 1, 2, scratch));
  CHECK_THROWS(read_tae_slice(
    reader, physical.size(), true, std::numeric_limits<std::uint64_t>::max(), 1, scratch));
  tae_read_at short_reader = [](auto, auto count, auto*) { return count - 1; };
  CHECK_THROWS(read_tae_slice(short_reader, physical.size(), true, 0, 1, scratch));
  CHECK_THROWS(read_tae_slice(short_reader, physical.size(), false, 0, 1, scratch));
}

TEST_CASE("a TAE compressed transport extent larger than 64 MiB uses bounded slices",
          "[tae_slices][cpu]")
{
  constexpr std::size_t length = (65u << 20) + 31;
  auto const physical_size     = length + ((length + 2043) / 2044) * 4;
  std::vector<std::uint8_t> scratch(tae_max_slice);
  std::size_t reads = 0, largest = 0;
  tae_read_at reader = [&](std::uint64_t offset, std::size_t count, std::uint8_t* target) {
    ++reads;
    largest = std::max(largest, count);
    for (std::size_t i = 0; i < count; ++i) {
      auto const physical = offset + i;
      auto const in_block = physical % 2048;
      target[i]           = in_block < 4 ? 0xa5 : (physical / 2048 * 2044 + in_block - 4) % 251;
    }
    return count;
  };
  std::size_t done = 0;
  while (done < length) {
    auto n = read_tae_slice(reader, physical_size, true, done, length - done, scratch);
    REQUIRE(n > 0);
    CHECK(scratch.front() == done % 251);
    CHECK(scratch[n / 2] == (done + n / 2) % 251);
    CHECK(scratch[n - 1] == (done + n - 1) % 251);
    done += n;
  }
  CHECK(done == length);
  CHECK(reads > 8);
  CHECK(largest <= tae_max_slice);
}
