/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "embedding/result_gpu.hpp"

#include "data/data_batch_utils.hpp"
#include "embedding/result_codec.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <map>
#include <set>

namespace sirius::embedding {
namespace {
struct publication {
  std::vector<cucascade::read_only_data_batch> batches;
  std::shared_ptr<terminal_ticket> ticket;
};
struct publisher_state final : terminal_admission,
                               capacity_waker,
                               std::enable_shared_from_this<publisher_state> {
  explicit publisher_state(std::size_t capacity) : limit(capacity) {}
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t const limit;
  uint64_t next{}, version{};
  bool sealed{}, producers_done{};
  std::set<uint64_t> live;
  std::map<uint64_t, std::unique_ptr<publication>> queue;
  std::vector<std::weak_ptr<capacity_waker>> subscribers;
  void wake() noexcept override
  {
    {
      std::lock_guard lock(mutex);
      ++version;
    }
    changed.notify_all();
  }
  void retire(uint64_t sequence) noexcept
  {
    std::array<std::shared_ptr<capacity_waker>, 128> notify;
    std::size_t count{};
    {
      std::lock_guard lock(mutex);
      live.erase(sequence);
      ++version;
      if (!sealed)
        for (auto const& weak : subscribers)
          if (auto wake = weak.lock()) notify[count++] = std::move(wake);
    }
    changed.notify_all();
    for (std::size_t i = 0; i < count; ++i)
      notify[i]->wake();
  }
  std::shared_ptr<terminal_ticket> try_acquire() override
  {
    auto ticket = std::make_shared<terminal_ticket>();
    uint64_t sequence;
    {
      std::lock_guard lock(mutex);
      if (sealed || live.size() == limit) return {};
      sequence = next++;
      live.insert(sequence);
    }
    auto weak        = weak_from_this();
    ticket->sequence = sequence;
    // The control-block failure path also invokes the deleter and retires the ID.
    ticket->credit = std::shared_ptr<void>(ticket.get(), [weak, sequence](void*) {
      if (auto state = weak.lock()) state->retire(sequence);
    });
    return ticket;
  }
  void subscribe(std::shared_ptr<capacity_waker> wake) override
  {
    std::lock_guard lock(mutex);
    if (sealed) return;
    for (auto const& existing : subscribers)
      if (existing.lock() == wake) return;
    if (subscribers.size() == 128)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED, "too many native terminal subscriptions");
    subscribers.emplace_back(wake);
  }
};
}  // namespace
struct result_publisher::impl {
  std::shared_ptr<native_result> result;
  std::vector<owned_column> schema;
  std::shared_ptr<publisher_state> state;
  clock::time_point deadline;
  std::function<void(std::exception_ptr)> failed;
  std::stop_callback<std::function<void()>> cancellation;
  std::thread worker;
  std::exception_ptr error;
  bool fatal{};
  std::unique_ptr<publication> current;
  std::shared_ptr<result_batch> scratch, filling;
  std::size_t payload_target{result_target};

  impl(std::shared_ptr<native_result> r,
       std::vector<owned_column> s,
       std::size_t limit,
       std::stop_token stop,
       clock::time_point end,
       std::function<void(std::exception_ptr)> fail)
    : result(std::move(r)),
      schema(std::move(s)),
      state(std::make_shared<publisher_state>(limit)),
      deadline(end),
      failed(std::move(fail)),
      cancellation(stop, [shared = state] {
        {
          std::lock_guard lock(shared->mutex);
          shared->sealed = true;
          ++shared->version;
        }
        shared->changed.notify_all();
      })
  {
    auto status = result->try_allocate(64u << 10, 0, scratch);
    if (status != SIRIUS_OK) throw failure(status, "native result scratch admission failed");
    auto available = result_window - scratch->credit.bytes();
    auto minimum   = result->allocation_charge(1, schema.size());
    if (minimum > available)
      throw failure(SIRIUS_RESOURCE_EXHAUSTED, "native pool blocks cannot fit result and scratch");
    // Prefer two independently owned batches, using the actual pool rounding
    // rather than assuming a particular pinned block size.
    auto capacity   = minimum <= available / 2 ? available / 2 : available;
    std::size_t low = 1, high = result_target;
    while (low < high) {
      auto middle = low + (high - low + 1) / 2;
      if (result->allocation_charge(middle, schema.size()) <= capacity)
        low = middle;
      else
        high = middle - 1;
    }
    payload_target = low;
    result->subscribe(state);
    worker = std::thread([this] { run(); });
  }
  void check()
  {
    if (clock::now() >= deadline)
      throw failure(SIRIUS_TIMEOUT, "native publication deadline expired");
    if (state->sealed) throw failure(SIRIUS_CANCELLED, "native publication cancelled");
  }
  void convert(cucascade::read_only_data_batch const& batch)
  {
    auto* space = batch.get_memory_space();
    if (!space || space->get_tier() != cucascade::memory::Tier::GPU)
      throw failure(SIRIUS_INVALID_STATE, "native terminal cursor lost GPU residency");
    if (cudaSetDevice(space->get_id().device_id) != cudaSuccess)
      throw pipeline::gpu_stream_quiescence_error("native publisher could not select GPU");
    rmm::cuda_stream stream;
    auto table = get_cudf_table_view(batch);
    uint32_t row{};
    while (row < static_cast<uint32_t>(table.num_rows())) {
      {
        std::lock_guard lock(state->mutex);
        check();
      }
      // Target includes room for actual allocator rounding and charged scratch.
      auto layout = size_native_result(table, row, schema, *scratch, stream.view(), payload_target);
      if (result->allocation_charge(layout.bytes, schema.size()) >
          result_window - scratch->credit.bytes())
        throw failure(SIRIUS_RESOURCE_EXHAUSTED,
                      "single native result row cannot fit the charged output window");
      while (true) {
        uint64_t version;
        {
          std::lock_guard lock(state->mutex);
          check();
          version = state->version;
        }
        auto status = result->try_allocate(layout.bytes, schema.size(), filling);
        if (status == SIRIUS_OK) break;
        if (status != SIRIUS_TIMEOUT)
          throw failure(status, "native result slice cannot be admitted");
        result->parked(true);
        struct parked_guard {
          native_result& result;
          ~parked_guard() { result.parked(false); }
        } parking{*result};
        {
          std::unique_lock lock(state->mutex);
          state->changed.wait_until(
            lock, deadline, [&] { return state->sealed || state->version != version; });
        }
      }
      encode_native_result(table, row, schema, layout, *filling, *scratch, stream.view());
      result->publish(std::move(filling));
      row += layout.rows;  // Commit cursor only after publication succeeds.
    }
  }
  void run() noexcept
  {
    try {
      while (true) {
        {
          std::unique_lock lock(state->mutex);
          state->changed.wait_until(lock, deadline, [&] {
            return state->sealed || (state->producers_done && state->live.empty()) ||
                   (!state->live.empty() && state->queue.contains(*state->live.begin()));
          });
          check();
          if (state->producers_done && state->live.empty()) break;
          auto first = state->queue.find(*state->live.begin());
          if (first == state->queue.end()) continue;
          current = std::move(first->second);
          state->queue.erase(first);
        }
        for (auto const& batch : current->batches)
          convert(batch);
        // Keep the ticket until the GPU owners have actually been released.
        current->batches.clear();
        current.reset();
      }
      scratch.reset();
    } catch (...) {
      error = std::current_exception();
      try {
        std::rethrow_exception(error);
      } catch (pipeline::gpu_stream_quiescence_error const&) {
        fatal = true;
      } catch (...) {
      }
      std::map<uint64_t, std::unique_ptr<publication>> discard;
      {
        std::lock_guard lock(state->mutex);
        state->sealed = true;
        state->subscribers.clear();
        if (!fatal) discard.swap(state->queue);
      }
      if (!fatal) {
        filling.reset();
        scratch.reset();
        current.reset();
      }
      failed(error);
      state->changed.notify_all();
    }
  }
};
result_publisher::result_publisher(std::shared_ptr<native_result> result,
                                   std::vector<owned_column> schema,
                                   std::size_t limit,
                                   std::stop_token stop,
                                   clock::time_point deadline,
                                   std::function<void(std::exception_ptr)> failed)
  : impl_(std::make_unique<impl>(
      std::move(result), std::move(schema), limit, stop, deadline, std::move(failed)))
{
}
result_publisher::~result_publisher()
{
  try {
    stop();
  } catch (...) {
  }
  if (impl_->fatal) (void)impl_.release();
}
std::shared_ptr<terminal_admission> result_publisher::admission() const { return impl_->state; }
void result_publisher::submit(op::operator_data const& input,
                              rmm::cuda_stream_view stream,
                              std::shared_ptr<terminal_ticket> ticket)
{
  if (!ticket) throw failure(SIRIUS_INVALID_STATE, "native result task has no admission");
  // A completed stream handoff makes the publisher independent of executor streams.
  if (cudaStreamSynchronize(stream.value()) != cudaSuccess)
    throw pipeline::gpu_stream_quiescence_error("native result handoff did not quiesce");
  auto value = std::make_unique<publication>();
  value->batches =
    dynamic_cast<op::pipelineable_operator_data const&>(input).get_read_only_batches();
  value->ticket = std::move(ticket);
  auto sequence = value->ticket->sequence;
  {
    std::lock_guard lock(impl_->state->mutex);
    impl_->check();
    if (!impl_->state->queue.emplace(sequence, std::move(value)).second)
      throw failure(SIRIUS_INVALID_STATE, "native terminal ticket published twice");
  }
  impl_->state->changed.notify_all();
}
void result_publisher::finish()
{
  {
    std::lock_guard lock(impl_->state->mutex);
    impl_->state->producers_done = true;
  }
  impl_->state->changed.notify_all();
  if (impl_->worker.joinable()) impl_->worker.join();
  if (impl_->error) std::rethrow_exception(impl_->error);
}
void result_publisher::stop()
{
  {
    std::lock_guard lock(impl_->state->mutex);
    impl_->state->sealed = true;
    impl_->state->subscribers.clear();
  }
  impl_->state->changed.notify_all();
  if (impl_->worker.joinable()) impl_->worker.join();
  if (impl_->fatal) std::rethrow_exception(impl_->error);
}
native_result_sink::native_result_sink(duckdb::vector<sirius::logical_type> types,
                                       std::size_t cardinality,
                                       std::shared_ptr<result_publisher> publisher)
  : sirius_physical_operator(
      op::SiriusPhysicalOperatorType::STREAMING_SINK, std::move(types), cardinality),
    publisher_(std::move(publisher))
{
}
void native_result_sink::build_pipelines(pipeline::sirius_pipeline& current,
                                         pipeline::sirius_meta_pipeline& meta)
{
  meta.get_state().add_pipeline_operator(current, *this);
  if (children.size() != 1)
    throw failure(SIRIUS_INVALID_STATE, "native result requires one producer");
  meta.create_child_meta_pipeline(current, *this).build(*children[0]);
}
std::unique_ptr<op::operator_data> native_result_sink::execute(op::operator_data const& input,
                                                               rmm::cuda_stream_view)
{
  return std::make_unique<op::pipelineable_operator_data>(
    dynamic_cast<op::pipelineable_operator_data const&>(input).get_read_only_batches());
}
std::shared_ptr<terminal_admission> native_result_sink::terminal_admission_control() const
{
  return publisher_->admission();
}
void native_result_sink::sink_admitted(op::operator_data const& input,
                                       rmm::cuda_stream_view stream,
                                       std::shared_ptr<terminal_ticket> ticket)
{
  publisher_->submit(input, stream, std::move(ticket));
}
}  // namespace sirius::embedding
