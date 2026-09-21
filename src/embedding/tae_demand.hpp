/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "embedding/buffer_budget.hpp"
#include "embedding/execution_stats.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace sirius::embedding {

inline constexpr std::size_t tae_host_capacity        = 64u << 20;
inline constexpr std::size_t tae_max_slice            = 8u << 20;
inline constexpr std::size_t tae_target_decoded_bytes = 32u << 20;
// This complete runtime envelope is acquired from the engine's EXISTING
// metadata budget once, when the first TAE read is registered. It is not an
// additional budget: a TAE query leaves at most 128 MiB of the default 256 MiB
// for its plan, manifest, schema and other reads.
inline constexpr std::size_t tae_metadata_blob_limit    = 8u << 20;
inline constexpr std::size_t tae_metadata_cache_limit   = 32u << 20;
inline constexpr std::size_t tae_work_metadata_total    = 64u << 20;
inline constexpr std::size_t tae_metadata_control_limit = 8u << 20;
inline constexpr std::size_t tae_metadata_reservation_bytes =
  tae_metadata_cache_limit + 3 * tae_metadata_blob_limit + tae_work_metadata_total +
  tae_metadata_control_limit;
inline constexpr std::size_t tae_work_metadata_max     = 8u << 20;
inline constexpr std::size_t tae_work_fixed_metadata   = 64u << 10;
inline constexpr std::size_t tae_chunk_metadata_charge = 4096;
inline constexpr std::size_t tae_chunk_vector_limit    = 128u << 10;
inline constexpr std::size_t tae_path_limit            = 4096;
inline constexpr std::size_t tae_pending_sources_limit = 64;
struct tae_demand_state;

// A permit covers queued, executing, retrying and retiring work. Its staging
// entitlement is charged now; actual pinned storage is allocated only after
// full GPU admission. The sum of entitlements fits the query's dedicated TAE
// budget, so no GPU worker ever waits for host capacity.
class tae_work_permit final {
 public:
  ~tae_work_permit();
  tae_work_permit(tae_work_permit const&)            = delete;
  tae_work_permit& operator=(tae_work_permit const&) = delete;
  std::size_t staging_bytes() const noexcept { return credit_.bytes(); }
  std::size_t metadata_bytes() const noexcept;
  std::size_t chunk_limit(std::size_t chunk_size) const noexcept;
  void record_payload_bytes(std::size_t bytes) const noexcept;
  // Only the admitted task installs storage. Its execution lease outlives a
  // replaced/destroyed scan input until stream retirement (including retry).
  void retain_staging(std::shared_ptr<void> storage) { storage_ = std::move(storage); }
  // Controller construction only; public for allocate_shared's constructor.
  tae_work_permit(buffer_budget::lease credit, std::shared_ptr<tae_demand_state> owner) noexcept;

 private:
  friend class tae_demand_controller;
  buffer_budget::lease credit_;
  std::shared_ptr<tae_demand_state> owner_;
  std::shared_ptr<void> storage_;
};

// One metadata worker per query, shared by every embedded TAE scan. request()
// only enqueues and must only be called for dependency-ready pipelines. The
// caller guarantees at most one outstanding request per source. Callbacks must
// catch and publish their own source errors. They run without the queue lock.
class tae_demand_controller final {
 public:
  using work = std::function<void(std::shared_ptr<tae_work_permit>)>;
  struct snapshot {
    std::size_t active, peak, queued, limit, slice_bytes, cached_metadata_bytes;
    bool closed;
  };

  tae_demand_controller(std::size_t gpu_streams,
                        std::shared_ptr<buffer_budget> host_budget,
                        std::size_t host_capacity              = tae_host_capacity,
                        std::shared_ptr<execution_stats> stats = {});
  ~tae_demand_controller();
  tae_demand_controller(tae_demand_controller const&)            = delete;
  tae_demand_controller& operator=(tae_demand_controller const&) = delete;
  bool request(work callback);
  void close() noexcept;
  snapshot inspect() const;

  // Only metadata-worker callbacks access this bounded LRU. Cached immutable
  // objects are not borrowed by descriptors: eviction never invalidates work.
  std::shared_ptr<const void> cached_metadata(std::string const& key);
  void cache_metadata(std::string key, std::shared_ptr<const void> value, std::size_t bytes);

 private:
  struct worker;
  std::shared_ptr<tae_demand_state> state_;
  std::unique_ptr<worker> worker_;
};

std::shared_ptr<tae_demand_controller> make_tae_demand_controller(
  std::size_t gpu_streams,
  std::shared_ptr<buffer_budget> host_budget,
  std::size_t host_capacity              = tae_host_capacity,
  std::shared_ptr<execution_stats> stats = {});
}  // namespace sirius::embedding
