/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/tae_demand.hpp"

#include "embedding/control.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sirius::embedding {
namespace {
constexpr std::size_t crc_block_bytes = 2048;
}  // namespace

struct tae_demand_state {
  struct cache_entry {
    std::string key;
    std::shared_ptr<const void> value;
    std::size_t bytes;
  };
  std::shared_ptr<buffer_budget> budget;
  std::shared_ptr<execution_stats> stats;
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<tae_demand_controller::work> pending;
  std::list<cache_entry> cache;
  // Controller slots include the worker's short admission/construction phase.
  // Public active/issued stats begin only after a real permit exists.
  std::size_t limit, slice, work_metadata, active{0}, peak{0}, cache_bytes{0};
  bool closed{false};
  void retire_slot() noexcept { retire(false); }
  void retire_permit() noexcept { retire(true); }
  void retire(bool issued) noexcept
  {
    {
      std::lock_guard lock(mutex);
      --active;
      if (issued && stats) stats->tae_complete();
    }
    changed.notify_one();
  }
};
struct tae_demand_controller::worker {
  std::thread thread;
};

tae_work_permit::tae_work_permit(buffer_budget::lease credit,
                                 std::shared_ptr<tae_demand_state> owner) noexcept
  : credit_(std::move(credit)), owner_(std::move(owner))
{
}
tae_work_permit::~tae_work_permit()
{
  // Restore the byte entitlement before making the work slot available.
  storage_.reset();
  credit_.reset();
  owner_->retire_permit();
}

std::size_t tae_work_permit::metadata_bytes() const noexcept { return owner_->work_metadata; }
void tae_work_permit::record_payload_bytes(std::size_t bytes) const noexcept
{
  if (owner_->stats) owner_->stats->tae_payload(bytes);
}
std::size_t tae_work_permit::chunk_limit(std::size_t chunk_size) const noexcept
{
  return std::min(tae_chunk_vector_limit / chunk_size,
                  (owner_->work_metadata - tae_work_fixed_metadata) / tae_chunk_metadata_charge);
}

tae_demand_controller::tae_demand_controller(std::size_t streams,
                                             std::shared_ptr<buffer_budget> budget,
                                             std::size_t capacity,
                                             std::shared_ptr<execution_stats> stats)
  : state_(std::make_shared<tae_demand_state>()), worker_(std::make_unique<worker>())
{
  if (!budget || streams > 128)
    throw std::invalid_argument("invalid TAE demand controller capacity");
  auto& s         = *state_;
  s.limit         = std::max<std::size_t>(2, streams * 2);
  s.work_metadata = std::min(tae_work_metadata_max, tae_work_metadata_total / s.limit);
  s.slice         = std::min(tae_max_slice, capacity / s.limit) / crc_block_bytes * crc_block_bytes;
  if (s.slice == 0 || capacity > tae_host_capacity)
    throw std::invalid_argument("TAE host capacity cannot provide a CRC slice per work permit");
  s.budget = std::move(budget);
  s.stats  = std::move(stats);
  if (s.stats) s.stats->tae_configure(s.limit, s.slice);
  worker_->thread = std::thread([state = state_] {
    for (;;) {
      work callback;
      {
        std::unique_lock lock(state->mutex);
        state->changed.wait(lock, [&] {
          return state->closed || (!state->pending.empty() && state->active < state->limit);
        });
        if (state->closed) return;
        callback = std::move(state->pending.front());
        state->pending.pop_front();
        ++state->active;
        state->peak = std::max(state->peak, state->active);
      }
      buffer_budget::lease credit;
      auto const status = state->budget->try_acquire(state->slice, credit);
      // Capacity is partitioned before work starts. Failure here indicates a
      // closed/miswired budget; deliver an empty permit so the source reports
      // an error without losing its durable wakeup.
      if (status != SIRIUS_OK) {
        state->retire_slot();
        callback(nullptr);
        continue;
      }
      std::shared_ptr<tae_work_permit> permit;
      try {
        permit = std::make_shared<tae_work_permit>(std::move(credit), state);
      } catch (...) {
        credit.reset();
        state->retire_slot();
        callback(nullptr);
        continue;
      }
      // No callback can retire this local permit before its issued transition.
      if (state->stats) {
        state->stats->tae_issue();
        state->stats->tae_staging(state->budget->inspect().peak);
      }
      callback(std::move(permit));
    }
  });
}

tae_demand_controller::~tae_demand_controller() { close(); }

bool tae_demand_controller::request(work callback)
{
  if (!callback) throw std::invalid_argument("empty TAE metadata callback");
  {
    std::lock_guard lock(state_->mutex);
    if (state_->closed) return false;
    if (state_->pending.size() == tae_pending_sources_limit)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED, "TAE pending metadata source limit reached");
    state_->pending.push_back(std::move(callback));
    if (state_->stats) state_->stats->tae_request(state_->pending.size());
  }
  state_->changed.notify_one();
  return true;
}

void tae_demand_controller::close() noexcept
{
  std::deque<work> discarded;
  {
    std::lock_guard lock(state_->mutex);
    state_->closed = true;
    discarded.swap(state_->pending);
  }
  state_->changed.notify_all();
  if (worker_->thread.joinable()) {
    // Normal query teardown owns the controller and joins here. If the last
    // source owner is released by its own callback, the worker's captured
    // shared state keeps the closed generation alive until the loop exits.
    if (worker_->thread.get_id() == std::this_thread::get_id())
      worker_->thread.detach();
    else
      worker_->thread.join();
  }
}

tae_demand_controller::snapshot tae_demand_controller::inspect() const
{
  std::lock_guard lock(state_->mutex);
  return {state_->active,
          state_->peak,
          state_->pending.size(),
          state_->limit,
          state_->slice,
          state_->cache_bytes,
          state_->closed};
}

std::shared_ptr<const void> tae_demand_controller::cached_metadata(std::string const& key)
{
  std::lock_guard lock(state_->mutex);
  for (auto it = state_->cache.begin(); it != state_->cache.end(); ++it) {
    if (it->key != key) continue;
    auto value = it->value;
    state_->cache.splice(state_->cache.begin(), state_->cache, it);
    return value;
  }
  return {};
}

void tae_demand_controller::cache_metadata(std::string key,
                                           std::shared_ptr<const void> value,
                                           std::size_t bytes)
{
  // Include both key storage and list/shared ownership bookkeeping; the parsed
  // object's vectors are charged by the caller before it is put in this LRU.
  static_assert(sizeof(tae_demand_state::cache_entry) + 4 * sizeof(void*) <= 512);
  if (key.size() > tae_path_limit + 32 || bytes > tae_metadata_cache_limit)
    throw std::runtime_error("TAE object metadata exceeds bounded cache capacity");
  bytes += 512 + 2 * key.size();
  if (bytes > tae_metadata_cache_limit)
    throw std::runtime_error("TAE object metadata exceeds bounded cache capacity");
  std::lock_guard lock(state_->mutex);
  for (auto it = state_->cache.begin(); it != state_->cache.end();) {
    if (it->key == key) {
      state_->cache_bytes -= it->bytes;
      it = state_->cache.erase(it);
    } else {
      ++it;
    }
  }
  while (state_->cache_bytes > tae_metadata_cache_limit - bytes) {
    state_->cache_bytes -= state_->cache.back().bytes;
    state_->cache.pop_back();
  }
  state_->cache.push_front({std::move(key), std::move(value), bytes});
  state_->cache_bytes += bytes;
  if (state_->stats) state_->stats->tae_cache(state_->cache_bytes);
}

std::shared_ptr<tae_demand_controller> make_tae_demand_controller(
  std::size_t streams,
  std::shared_ptr<buffer_budget> budget,
  std::size_t capacity,
  std::shared_ptr<execution_stats> stats)
{
  return std::make_shared<tae_demand_controller>(
    streams, std::move(budget), capacity, std::move(stats));
}
}  // namespace sirius::embedding
