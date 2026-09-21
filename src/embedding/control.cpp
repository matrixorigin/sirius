/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding/control.hpp"

#include "embedding/execution_interrupted.hpp"
#include "embedding/input.hpp"
#include "embedding/result.hpp"
#include "embedding/tae_demand.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <new>
#include <utility>

namespace sirius::embedding {
namespace {
sirius_error error(sirius_status code, const char* message = "") noexcept
{
  sirius_error result{};
  assign_error(result, code, message);
  return result;
}
bool quiesced(query_state const& q)
{
  return q.phase == query_phase::QUIESCED || q.phase == query_phase::CLOSED;
}
bool terminal(query_state const& q) { return quiesced(q) || q.phase == query_phase::UNAVAILABLE; }
}  // namespace
void assign_error(sirius_error& out, sirius_status code, const char* message) noexcept
{
  out.code = code;
  std::snprintf(out.message, sizeof(out.message), "%s", message ? message : "");
}
sirius_error current_error() noexcept
{
  try {
    throw;
  } catch (failure const& e) {
    return e.error;
  } catch (execution_interrupted const& e) {
    return error(e.deadline_expired ? SIRIUS_TIMEOUT : SIRIUS_CANCELLED, e.what());
  } catch (pipeline::gpu_stream_quiescence_error const& e) {
    return error(SIRIUS_GPU_UNAVAILABLE, e.what());
  } catch (std::bad_alloc const&) {
    return error(SIRIUS_RESOURCE_EXHAUSTED, "native allocation failed");
  } catch (std::exception const& e) {
    return error(SIRIUS_EXECUTION_FAILED, e.what());
  } catch (...) {
    return error(SIRIUS_EXECUTION_FAILED, "unknown native failure");
  }
}

engine_control::engine_control(factory create_backend, std::size_t max_waiting)
  : create_backend_(std::move(create_backend)), max_waiting_(max_waiting)
{
  if (!create_backend_ || max_waiting == 0 || max_waiting > 16)
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid native coordinator configuration");
}
engine_control::~engine_control() { assert(!worker_.joinable()); }
sirius_error engine_control::initialize()
{
  worker_ = std::thread([this] { worker(); });
  std::unique_lock lock(mutex_);
  changed_.wait(lock, [&] { return initialized_; });
  auto result = initialization_error_;
  lock.unlock();
  if (result.code != SIRIUS_OK) worker_.join();
  return result;
}
std::shared_ptr<query_state> engine_control::create(std::string_view plan,
                                                    std::chrono::milliseconds timeout)
{
  if (plan.empty() || plan.size() > (16u << 20) || timeout.count() <= 0)
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid query plan or deadline");
  std::lock_guard lock(mutex_);
  if (unavailable_) throw failure(SIRIUS_GPU_UNAVAILABLE, "native runtime is unavailable");
  if (!initialized_ || initialization_error_.code != SIRIUS_OK)
    throw failure(SIRIUS_INVALID_STATE, "native runtime is not initialized");
  if (!accepting_) throw failure(SIRIUS_INVALID_STATE, "native runtime is stopping");
  if (live_ == max_waiting_ + 1)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native query capacity reached");
  auto metadata_charge = plan.size() * 4;
  buffer_budget::lease plan_credit;
  auto credit_status = metadata_budget_->acquire(metadata_charge, {}, clock::now(), plan_credit);
  if (credit_status != SIRIUS_OK)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native metadata capacity reached");
  // Admission precedes the bounded copy; rejected concurrent callers do not
  // each allocate a maximum-size plan outside the query-count limit.
  auto q            = std::make_shared<query_state>();
  q->inputs         = std::make_shared<input_registry>();
  q->results        = std::make_shared<native_result>();
  q->plan           = plan;
  q->metadata_bytes = metadata_charge;
  q->metadata_leases.push_back(std::move(plan_credit));
  q->deadline = clock::now() + timeout;
  for (std::size_t i = 0; i < queries_.size(); ++i) {
    if (queries_[i].expired()) {
      q->slot     = i;
      queries_[i] = q;
      ++live_;
      return q;
    }
  }
  throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native query handles exhausted");
}
namespace {
std::string copy_text(const char* data, std::size_t bytes, std::size_t limit, const char* message)
{
  if ((!data && bytes) || bytes > limit) throw failure(SIRIUS_INVALID_ARGUMENT, message);
  std::string result(data ? data : "", bytes);
  if (result.find('\0') != std::string::npos) throw failure(SIRIUS_INVALID_ARGUMENT, message);
  return result;
}
owned_column copy_column(sirius_column const& c, std::size_t& bytes)
{
  if (c.reserved || c.nullable > 1) throw failure(SIRIUS_INVALID_ARGUMENT, "invalid column");
  sirius_input_column scalar{c.oid, c.width, c.scale, c.nullable};
  validate_input_schema({&scalar, 1});
  owned_column out{c.oid,
                   c.width,
                   c.scale,
                   c.nullable != 0,
                   copy_text(c.name, c.name_bytes, 1u << 20, "invalid column name")};
  bytes += sizeof(owned_column) + out.name.size();
  return out;
}
}  // namespace
void engine_control::bind_query(std::shared_ptr<query_state> const& q,
                                sirius_query_contract const& contract)
{
  if (contract.struct_size != sizeof(contract) || contract.abi_version != SIRIUS_ABI_VERSION ||
      contract.reserved || contract.query_id_bytes == 0 || contract.query_id_bytes > 4096 ||
      contract.output_column_count > input_columns_limit ||
      (contract.output_column_count && !contract.output_columns))
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid native query contract");
  std::size_t output_names = 0;
  for (uint32_t i = 0; i < contract.output_column_count; ++i) {
    auto const& column = contract.output_columns[i];
    if (column.name_bytes == 0 || column.name_bytes > (1u << 20) ||
        output_names > (1u << 20) - column.name_bytes)
      throw failure(SIRIUS_INVALID_ARGUMENT, "output names exceed limit");
    output_names += column.name_bytes;
    sirius_input_column scalar{column.oid, column.width, column.scale, column.nullable};
    validate_input_schema({&scalar, 1});
  }
  auto canonical = sizeof(owned_query_contract) + contract.query_id_bytes +
                   contract.output_column_count * sizeof(owned_column) + output_names;
  auto charged = canonical * 4;
  buffer_budget::lease credit;
  auto credit_status = metadata_budget_->acquire(charged, {}, clock::now(), credit);
  if (credit_status != SIRIUS_OK)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native metadata capacity reached");
  auto owned        = std::make_unique<owned_query_contract>();
  owned->account_id = contract.account_id;
  owned->query_id =
    copy_text(contract.query_id, contract.query_id_bytes, 4096, "invalid native query identifier");
  std::copy(
    std::begin(contract.snapshot_ts), std::end(contract.snapshot_ts), owned->snapshot_ts.begin());
  std::size_t actual_charge = sizeof(owned_query_contract) + owned->query_id.size();
  owned->outputs.reserve(contract.output_column_count);
  for (uint32_t i = 0; i < contract.output_column_count; ++i) {
    owned->outputs.push_back(copy_column(contract.output_columns[i], actual_charge));
  }
  if (actual_charge > canonical)
    throw failure(SIRIUS_INVALID_ARGUMENT, "output contract size overflow");
  std::vector<sirius_column> schema;
  schema.reserve(owned->outputs.size());
  for (auto const& c : owned->outputs)
    schema.push_back({c.oid,
                      c.width,
                      c.scale,
                      c.nullable,
                      c.name.data(),
                      static_cast<uint32_t>(c.name.size()),
                      0});
  std::lock_guard lock(mutex_);
  if (q->phase != query_phase::CREATED || q->contract)
    throw failure(SIRIUS_INVALID_STATE, "query binding requires one created query");
  q->metadata_leases.reserve(q->metadata_leases.size() + 1);
  q->metadata_bytes += charged;
  q->metadata_leases.push_back(std::move(credit));
  q->contract      = std::move(owned);
  q->result_schema = std::move(schema);
}
void engine_control::require_result_active(std::shared_ptr<query_state> const& q)
{
  std::lock_guard lock(mutex_);
  if (!q->started || q->phase == query_phase::CLOSED)
    throw failure(SIRIUS_INVALID_STATE, "native result requires a started query");
}
sirius_result_schema engine_control::result_schema(std::shared_ptr<query_state> const& q)
{
  std::lock_guard lock(mutex_);
  if (!q->contract || q->phase == query_phase::CREATED || q->phase == query_phase::QUEUED ||
      q->phase == query_phase::PREPARING || q->phase == query_phase::CLOSED)
    throw failure(SIRIUS_INVALID_STATE, "result schema requires prepared query");
  return {sizeof(sirius_result_schema),
          SIRIUS_ABI_VERSION,
          static_cast<uint32_t>(q->result_schema.size()),
          0,
          q->result_schema.data()};
}
void engine_control::register_read(std::shared_ptr<query_state> const& q,
                                   sirius_read_binding const& binding)
{
  if (binding.struct_size != sizeof(binding) || binding.abi_version != SIRIUS_ABI_VERSION ||
      binding.reserved || binding.binding_id == 0 ||
      binding.binding_id > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      (binding.source_kind != SIRIUS_READ_MO && binding.source_kind != SIRIUS_READ_TAE) ||
      binding.column_count > input_columns_limit || (binding.column_count && !binding.columns) ||
      binding.tae_manifest_bytes > (64u << 20))
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid native read binding");
  auto add_size = [](std::size_t& total, std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - total)
      throw failure(SIRIUS_INVALID_ARGUMENT, "native read metadata size overflow");
    total += value;
  };
  for (auto const& identity : {std::pair{binding.database_name, binding.database_name_bytes},
                               std::pair{binding.table_name, binding.table_name_bytes},
                               std::pair{binding.schema_name, binding.schema_name_bytes},
                               std::pair{binding.data_root, binding.data_root_bytes}}) {
    if (identity.second > (1u << 20) || (!identity.first && identity.second))
      throw failure(SIRIUS_INVALID_ARGUMENT, "invalid native read identity");
  }
  std::size_t canonical = sizeof(owned_read_binding);
  add_size(canonical, binding.database_name_bytes);
  add_size(canonical, binding.table_name_bytes);
  add_size(canonical, binding.schema_name_bytes);
  add_size(canonical, binding.data_root_bytes);
  add_size(canonical, binding.column_count * sizeof(owned_read_column));
  for (uint32_t i = 0; i < binding.column_count; ++i) {
    auto const& column = binding.columns[i];
    if (column.reserved || column.sequence_number > UINT16_MAX || column.logical.name_bytes == 0 ||
        column.logical.name_bytes > (1u << 20))
      throw failure(SIRIUS_INVALID_ARGUMENT, "invalid physical column binding");
    sirius_input_column scalar{
      column.logical.oid, column.logical.width, column.logical.scale, column.logical.nullable};
    validate_input_schema({&scalar, 1});
    add_size(canonical, column.logical.name_bytes);
  }
  if (canonical > (1u << 20))
    throw failure(SIRIUS_INVALID_ARGUMENT, "canonical read metadata exceeds 1 MiB");
  auto manifest_bytes = static_cast<std::size_t>(binding.tae_manifest_bytes);
  if (manifest_bytes > (std::numeric_limits<std::size_t>::max() - canonical) / 8)
    throw failure(SIRIUS_INVALID_ARGUMENT, "TAE metadata size overflow");
  auto charged = canonical * 4 + manifest_bytes * 8;
  buffer_budget::lease credit;
  auto credit_status = metadata_budget_->acquire(charged, {}, clock::now(), credit);
  if (credit_status != SIRIUS_OK)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native metadata capacity reached");

  owned_read_binding owned;
  owned.binding_id    = binding.binding_id;
  owned.source_kind   = binding.source_kind;
  owned.database_name = copy_text(
    binding.database_name, binding.database_name_bytes, 1u << 20, "invalid database name");
  owned.table_name =
    copy_text(binding.table_name, binding.table_name_bytes, 1u << 20, "invalid table name");
  owned.schema_name =
    copy_text(binding.schema_name, binding.schema_name_bytes, 1u << 20, "invalid schema name");
  owned.data_root =
    copy_text(binding.data_root, binding.data_root_bytes, 1u << 20, "invalid TAE data root");
  if (binding.tae_manifest_bytes && !binding.tae_manifest)
    throw failure(SIRIUS_INVALID_ARGUMENT, "missing TAE manifest bytes");
  if (binding.tae_manifest_bytes)
    owned.manifest.assign(static_cast<const char*>(binding.tae_manifest),
                          static_cast<std::size_t>(binding.tae_manifest_bytes));
  if (owned.source_kind == SIRIUS_READ_MO && (!owned.manifest.empty() || !owned.data_root.empty()))
    throw failure(SIRIUS_INVALID_ARGUMENT, "MO binding cannot contain TAE metadata");
  if (owned.source_kind == SIRIUS_READ_TAE && (owned.manifest.empty() || owned.data_root.empty()))
    throw failure(SIRIUS_INVALID_ARGUMENT, "TAE binding requires manifest bytes and data root");
  owned.columns.reserve(binding.column_count);
  for (uint32_t i = 0; i < binding.column_count; ++i) {
    auto const& c            = binding.columns[i];
    std::size_t copied_bytes = 0;
    owned.columns.push_back(
      {copy_column(c.logical, copied_bytes), c.physical_column_id, c.sequence_number});
  }
  std::lock_guard lock(mutex_);
  if (q->phase != query_phase::CREATED)
    throw failure(SIRIUS_INVALID_STATE, "read registration requires a created query");
  if (q->bindings.size() == 16)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native query read limit reached");
  for (auto const& current : q->bindings)
    if (current.binding_id == owned.binding_id)
      throw failure(SIRIUS_INVALID_ARGUMENT, "duplicate native read binding");
  // Reserve the complete bounded TAE runtime envelope BEFORE prepare can
  // allocate its cache, object metadata or work descriptors. Serialized plan
  // and manifest accounting above remains additive within the same 256 MiB
  // engine budget. Registration is serialized so two first reads cannot both
  // omit the charge (or both retain it).
  buffer_budget::lease tae_runtime_credit;
  bool const first_tae = owned.source_kind == SIRIUS_READ_TAE &&
                         std::none_of(q->bindings.begin(), q->bindings.end(), [](auto const& read) {
                           return read.source_kind == SIRIUS_READ_TAE;
                         });
  if (first_tae && metadata_budget_->try_acquire(tae_metadata_reservation_bytes,
                                                 tae_runtime_credit) != SIRIUS_OK)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native TAE runtime metadata capacity reached");
  q->metadata_leases.reserve(q->metadata_leases.size() + (first_tae ? 2 : 1));
  q->bindings.reserve(q->bindings.size() + 1);
  q->metadata_bytes += charged + (first_tae ? tae_metadata_reservation_bytes : 0);
  q->metadata_leases.push_back(std::move(credit));
  if (first_tae) q->metadata_leases.push_back(std::move(tae_runtime_credit));
  q->bindings.push_back(std::move(owned));
}
sirius_error engine_control::prepare(std::shared_ptr<query_state> const& q,
                                     std::chrono::milliseconds duration)
{
  std::unique_lock lock(mutex_);
  if (q->phase == query_phase::PREPARED || terminal(*q)) return q->result;
  if (q->phase != query_phase::CREATED && q->phase != query_phase::QUEUED &&
      q->phase != query_phase::PREPARING)
    return error(SIRIUS_INVALID_STATE, "query preparation already completed");
  if (q->phase == query_phase::CREATED) {
    if (!accepting_) return error(SIRIUS_INVALID_STATE, "native runtime is stopping");
    if (pending_.size() >= max_waiting_ + (active_ ? 0 : 1))
      return error(SIRIUS_RESOURCE_EXHAUSTED, "native preparation queue is full");
    pending_.push_back(q);
    q->phase = query_phase::QUEUED;
    changed_.notify_all();
  }
  if (!changed_.wait_for(
        lock, duration, [&] { return q->phase == query_phase::PREPARED || terminal(*q); }))
    return error(SIRIUS_TIMEOUT, "waiting for native preparation timed out");
  return q->result;
}
std::shared_ptr<native_input> engine_control::register_input(std::shared_ptr<query_state> const& q,
                                                             uint64_t id,
                                                             const sirius_input_column* columns,
                                                             uint32_t count)
{
  if (count > input_columns_limit || (count && !columns))
    throw failure(SIRIUS_INVALID_ARGUMENT, "invalid native input schema");
  std::lock_guard lock(mutex_);
  if (q->phase != query_phase::CREATED || !accepting_ || q->stop.stop_requested())
    throw failure(SIRIUS_INVALID_STATE, "input registration requires a created query");
  if (q->inputs->reads.size() == 16)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native query read limit reached");
  for (auto const& read : q->inputs->reads)
    if (read->id == id) throw failure(SIRIUS_INVALID_ARGUMENT, "duplicate native input binding");
  std::size_t metadata_bytes = sizeof(native_input) + sizeof(std::shared_ptr<native_input>) +
                               static_cast<std::size_t>(count) * sizeof(sirius_input_column);
  buffer_budget::lease metadata_credit;
  if (metadata_budget_->acquire(metadata_bytes, {}, clock::now(), metadata_credit) != SIRIUS_OK)
    throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native metadata capacity reached");
  q->metadata_leases.reserve(q->metadata_leases.size() + 1);
  q->inputs->reads.reserve(q->inputs->reads.size() + 1);
  std::vector<sirius_input_column> schema;
  if (count) schema.assign(columns, columns + count);
  auto read =
    std::make_shared<native_input>(id, std::move(schema), q->stop.get_token(), q->deadline);
  q->inputs->reads.push_back(read);
  q->metadata_bytes += metadata_bytes;
  q->metadata_leases.push_back(std::move(metadata_credit));
  ++q->inputs->handles;
  return read;
}
void engine_control::require_input_active(std::shared_ptr<query_state> const& q)
{
  std::lock_guard lock(mutex_);
  if ((q->phase != query_phase::PREPARED && q->phase != query_phase::RUNNING) || !q->startable)
    throw failure(SIRIUS_INVALID_STATE, "native input requires successful query preparation");
}
sirius_error engine_control::start(std::shared_ptr<query_state> const& q)
{
  std::lock_guard lock(mutex_);
  if (q->phase != query_phase::PREPARED || q->started || q->stop.stop_requested())
    return error(SIRIUS_INVALID_STATE, "query is not startable");
  if (!q->startable) return error(SIRIUS_UNSUPPORTED, "native result adapter is not installed");
  q->started = true;
  changed_.notify_all();
  return {};
}
void engine_control::cancel(std::shared_ptr<query_state> const& q)
{
  // stop callbacks must be nonblocking; invoking one under mutex_ could deadlock.
  q->stop.request_stop();
  q->results->cancel();
  {
    std::lock_guard lock(mutex_);
    if (q->phase == query_phase::CREATED || q->phase == query_phase::QUEUED) {
      std::erase(pending_, q);
      q->result = error(SIRIUS_CANCELLED, "native query cancelled before preparation");
      q->phase  = query_phase::QUIESCED;
    }
  }
  changed_.notify_all();
}
sirius_error engine_control::wait(std::shared_ptr<query_state> const& q,
                                  std::chrono::milliseconds duration)
{
  std::unique_lock lock(mutex_);
  if (!changed_.wait_for(lock, duration, [&] { return terminal(*q); }))
    return error(SIRIUS_TIMEOUT, "waiting for native quiescence timed out");
  return q->result;
}
sirius_error engine_control::close_query(std::shared_ptr<query_state> const& q,
                                         std::chrono::milliseconds duration)
{
  cancel(q);
  auto result = wait(q, duration);
  // TIMEOUT may also be the execution outcome. Inspect the state rather than
  // confusing a quiesced expired query with a still-running wait timeout.
  std::lock_guard lock(mutex_);
  if (!quiesced(*q) || result.code == SIRIUS_GPU_UNAVAILABLE) return result;
  if (q->inputs->handles || q->inputs->filling_handles)
    return error(SIRIUS_BUSY, "close outstanding native input handles and leases first");
  if (q->results->borrowed() || q->results->inspect().leases != 0)
    return error(SIRIUS_BUSY, "release outstanding or retiring native result batches first");
  if (q->phase != query_phase::CLOSED) {
    // Keep the CPU facades valid for an engine-stop snapshot already holding q,
    // but release runtime-dependent owners before allowing backend retirement.
    q->inputs->reads.clear();
    q->metadata_bytes = 0;
    q->metadata_leases.clear();
    q->plan.clear();
    q->contract.reset();
    q->result_schema.clear();
    q->bindings.clear();
    q->phase = query_phase::CLOSED;
    queries_[q->slot].reset();
    --live_;
  }
  return {};
}
void engine_control::stop()
{
  std::array<std::shared_ptr<query_state>, 17> queries;
  {
    std::lock_guard lock(mutex_);
    accepting_ = false;
    for (std::size_t i = 0; i < queries_.size(); ++i)
      queries[i] = queries_[i].lock();
  }
  for (auto const& q : queries)
    if (q) cancel(q);
  changed_.notify_all();
}
sirius_error engine_control::close(std::chrono::milliseconds duration)
{
  stop();
  std::unique_lock lock(mutex_);
  if (unavailable_) return error(SIRIUS_GPU_UNAVAILABLE, "native runtime requires process restart");
  if (live_ != 0) return error(SIRIUS_BUSY, "close outstanding native query handles first");
  exit_requested_ = true;
  changed_.notify_all();
  if (!changed_.wait_for(lock, duration, [&] { return exited_; }))
    return error(SIRIUS_TIMEOUT, "native runtime shutdown timed out");
  lock.unlock();
  if (worker_.joinable()) worker_.join();
  return {};
}
sirius_engine_stats engine_control::inspect()
{
  std::lock_guard lock(mutex_);
  return {sizeof(sirius_engine_stats),
          SIRIUS_ABI_VERSION,
          accepting_,
          unavailable_,
          live_,
          pending_.size()};
}
void engine_control::process(engine_backend& backend,
                             std::shared_ptr<query_state> const& q) noexcept
{
  std::unique_ptr<query_driver> driver;
  sirius_error result{};
  try {
    if (q->stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "native query cancelled");
    if (clock::now() >= q->deadline) throw failure(SIRIUS_TIMEOUT, "native query deadline expired");
    driver = backend.prepare_bound(q->plan, *q, *q->inputs, q->stop.get_token(), q->deadline);
    if (!driver) throw failure(SIRIUS_EXECUTION_FAILED, "native backend returned no query owner");
    {
      std::unique_lock lock(mutex_);
      q->phase     = query_phase::PREPARED;
      q->startable = driver->startable();
      changed_.notify_all();
      if (!changed_.wait_until(
            lock, q->deadline, [&] { return q->started || q->stop.stop_requested(); }) ||
          clock::now() >= q->deadline)
        throw failure(SIRIUS_TIMEOUT, "unstarted native query expired");
      if (q->stop.stop_requested()) throw failure(SIRIUS_CANCELLED, "native query cancelled");
      q->phase = query_phase::RUNNING;
    }
    driver->run(q->stop.get_token(), q->deadline);
    if (q->stop.stop_requested())
      result = error(SIRIUS_CANCELLED, "native query cancelled");
    else if (clock::now() >= q->deadline)
      result = error(SIRIUS_TIMEOUT, "native query deadline expired");
  } catch (...) {
    result = current_error();
  }
  if (!backend.available())
    result = error(SIRIUS_GPU_UNAVAILABLE, "native backend health is unavailable");
  {
    std::lock_guard lock(mutex_);
    q->phase = query_phase::DRAINING;
  }
  if (driver && result.code != SIRIUS_GPU_UNAVAILABLE) {
    try {
      driver->finish();
    } catch (...) {
      result = error(SIRIUS_GPU_UNAVAILABLE, "native cleanup could not prove quiescence");
    }
  }
  if (!backend.available())
    result = error(SIRIUS_GPU_UNAVAILABLE, "native backend cleanup poisoned the runtime");
  if (result.code == SIRIUS_GPU_UNAVAILABLE) {
    // The process is now the cleanup owner. Do not destruct live CUDA buffers
    // or a thread-affine window after a failed synchronization.
    (void)driver.release();
  } else {
    driver.reset();
    auto producer_error = q->inputs->outcome();
    if (producer_error.code) result = producer_error;
    q->inputs->stop();
    q->inputs->discard();
  }
  q->results->complete(result);
  {
    std::lock_guard lock(mutex_);
    q->result = result;
    q->phase =
      result.code == SIRIUS_GPU_UNAVAILABLE ? query_phase::UNAVAILABLE : query_phase::QUIESCED;
    active_.reset();
    if (result.code == SIRIUS_GPU_UNAVAILABLE) {
      unavailable_ = true;
      accepting_   = false;
    }
  }
  if (result.code == SIRIUS_GPU_UNAVAILABLE) stop();
  changed_.notify_all();
}
void engine_control::worker() noexcept
{
  std::unique_ptr<engine_backend> backend;
  try {
    backend = create_backend_();
    if (!backend) throw failure(SIRIUS_EXECUTION_FAILED, "native backend initialization failed");
    auto metadata_capacity = backend->metadata_capacity_bytes();
    if (!metadata_capacity)
      throw failure(SIRIUS_INVALID_ARGUMENT, "native metadata capacity must be positive");
    metadata_budget_ = std::make_unique<buffer_budget>(metadata_capacity, 1024);
  } catch (...) {
    std::lock_guard lock(mutex_);
    initialization_error_ = current_error();
    initialized_ = exited_ = true;
    accepting_             = false;
    changed_.notify_all();
    return;
  }
  {
    std::lock_guard lock(mutex_);
    initialized_ = true;
    changed_.notify_all();
  }
  for (;;) {
    std::shared_ptr<query_state> q;
    {
      std::unique_lock lock(mutex_);
      changed_.wait(lock, [&] { return exit_requested_ || !pending_.empty(); });
      if (exit_requested_) break;
      q = std::move(pending_.front());
      pending_.pop_front();
      q->phase = query_phase::PREPARING;
      active_  = q;
    }
    process(*backend, q);
  }
  backend.reset();
  std::lock_guard lock(mutex_);
  exited_ = true;
  changed_.notify_all();
}
}  // namespace sirius::embedding
