/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding/control.hpp"
#include "embedding/input.hpp"
#include "sirius_c.h"

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

using sirius::embedding::engine_control;
using sirius::embedding::failure;
using sirius::embedding::query_state;

struct sirius_engine_handle {
  std::shared_ptr<engine_control> control;
};
struct sirius_query_handle {
  std::shared_ptr<engine_control> control;
  std::shared_ptr<query_state> state;
};
struct sirius_input_handle {
  std::shared_ptr<engine_control> control;
  std::shared_ptr<query_state> query;
  std::shared_ptr<sirius::embedding::native_input> input;
};
struct sirius_batch_handle {
  std::shared_ptr<query_state> query;
  std::shared_ptr<sirius::embedding::input_batch> batch;
  ~sirius_batch_handle()
  {
    if (batch) {
      batch.reset();
      --query->inputs->filling_handles;
    }
  }
};

namespace {
template <typename Fn>
sirius_status boundary(sirius_error* out, Fn&& fn) noexcept
{
  sirius_error result{};
  try {
    result = fn();
  } catch (...) {
    result = sirius::embedding::current_error();
  }
  if (out) *out = result;
  return result.code;
}
void require(bool valid, const char* message)
{
  if (!valid) throw failure(SIRIUS_INVALID_ARGUMENT, message);
}
template <typename T>
void require_options(T const* options)
{
  require(
    options && options->struct_size == sizeof(T) && options->abi_version == SIRIUS_ABI_VERSION,
    "incompatible native options layout or ABI version");
}
}  // namespace

extern "C" uint32_t sirius_abi_version(void) { return SIRIUS_ABI_VERSION; }
extern "C" uint64_t sirius_capabilities(void) { return SIRIUS_CAP_ENGINE_CONTROL; }

extern "C" sirius_status sirius_engine_create(const sirius_engine_options* options,
                                              sirius_engine_handle** out,
                                              sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(out != nullptr && *out == nullptr, "engine output must be an empty handle");
    require_options(options);
    require(
      options->config_path && options->config_path_bytes > 0 && options->config_path_bytes <= 4096,
      "an explicit bounded native configuration path is required");
    auto config = std::string(options->config_path, options->config_path_bytes);
    require(config.find('\0') == std::string::npos, "configuration path contains a NUL byte");
    auto waiting = options->max_waiting_queries == 0 ? 16 : options->max_waiting_queries;
    auto streams = options->gpu_streams == 0 ? 2 : options->gpu_streams;
    require(streams <= 128 && options->reserved == 0,
            "invalid native GPU worker count or option flags");
    auto engine     = std::make_unique<sirius_engine_handle>();
    engine->control = std::make_shared<engine_control>(
      sirius::embedding::native_backend_factory(std::move(config), streams), waiting);
    auto status = engine->control->initialize();
    if (status.code != SIRIUS_OK) return status;
    *out = engine.release();
    return {};
  });
}
extern "C" sirius_status sirius_engine_get_stats(sirius_engine_handle* engine,
                                                 sirius_engine_stats* out,
                                                 sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(engine && out, "missing engine or stats output");
    require_options(out);
    *out = engine->control->inspect();
    return {};
  });
}
extern "C" sirius_status sirius_engine_stop(sirius_engine_handle* engine, sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(engine != nullptr, "missing engine");
    engine->control->stop();
    return {};
  });
}
extern "C" sirius_status sirius_engine_close(sirius_engine_handle** engine,
                                             uint32_t wait_ms,
                                             sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(engine != nullptr, "missing engine handle address");
    if (!*engine) return {};
    auto status = (*engine)->control->close(std::chrono::milliseconds(wait_ms));
    if (status.code == SIRIUS_OK) {
      delete *engine;
      *engine = nullptr;
    }
    return status;
  });
}
extern "C" sirius_status sirius_query_create(sirius_engine_handle* engine,
                                             const sirius_query_options* options,
                                             const void* plan,
                                             uint64_t plan_bytes,
                                             sirius_query_handle** out,
                                             sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(engine && out && *out == nullptr && plan && plan_bytes > 0 && plan_bytes <= (16u << 20),
            "invalid query plan or output");
    require_options(options);
    require(options->reserved == 0, "unknown query option flags");
    auto query     = std::make_unique<sirius_query_handle>();
    query->control = engine->control;
    query->state   = query->control->create(
      std::string_view(static_cast<const char*>(plan), static_cast<std::size_t>(plan_bytes)),
      std::chrono::milliseconds(options->timeout_ms == 0 ? 900000 : options->timeout_ms));
    *out = query.release();
    return {};
  });
}
extern "C" sirius_status sirius_query_prepare(sirius_query_handle* query,
                                              uint32_t wait_ms,
                                              sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(query != nullptr, "missing query");
    return query->control->prepare(query->state, std::chrono::milliseconds(wait_ms));
  });
}
extern "C" sirius_status sirius_query_start(sirius_query_handle* query, sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(query != nullptr, "missing query");
    return query->control->start(query->state);
  });
}
extern "C" sirius_status sirius_query_cancel(sirius_query_handle* query, sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(query != nullptr, "missing query");
    query->control->cancel(query->state);
    return {};
  });
}
extern "C" sirius_status sirius_query_wait(sirius_query_handle* query,
                                           uint32_t wait_ms,
                                           sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(query != nullptr, "missing query");
    return query->control->wait(query->state, std::chrono::milliseconds(wait_ms));
  });
}
extern "C" sirius_status sirius_query_close(sirius_query_handle** query,
                                            uint32_t wait_ms,
                                            sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(query != nullptr, "missing query handle address");
    if (!*query) return {};
    auto status =
      (*query)->control->close_query((*query)->state, std::chrono::milliseconds(wait_ms));
    if (status.code == SIRIUS_OK) {
      delete *query;
      *query = nullptr;
    }
    return status;
  });
}

extern "C" sirius_status sirius_input_register(sirius_query_handle* query,
                                               uint64_t id,
                                               const sirius_input_column* columns,
                                               uint32_t count,
                                               sirius_input_handle** out,
                                               sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(query && out && !*out, "invalid native input output");
    auto handle     = std::make_unique<sirius_input_handle>();
    handle->control = query->control;
    handle->query   = query->state;
    handle->input   = query->control->register_input(query->state, id, columns, count);
    *out            = handle.release();
    return {};
  });
}
extern "C" sirius_status sirius_input_acquire(sirius_input_handle* input,
                                              uint64_t bytes,
                                              uint32_t wait_ms,
                                              sirius_batch_handle** out,
                                              sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(input && out && !*out && bytes <= sirius::embedding::input_window,
            "invalid native batch output or size");
    auto outcome = input->input->outcome();
    if (outcome.code) return outcome;
    input->control->require_input_active(input->query);
    auto handle   = std::make_unique<sirius_batch_handle>();
    handle->query = input->query;
    handle->batch = input->input->acquire(
      bytes, sirius::embedding::clock::now() + std::chrono::milliseconds(wait_ms));
    ++handle->query->inputs->filling_handles;
    *out = handle.release();
    return {};
  });
}
extern "C" sirius_status sirius_input_write(sirius_batch_handle* batch,
                                            uint64_t offset,
                                            const void* data,
                                            uint64_t bytes,
                                            sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(batch && batch->batch && (data || !bytes), "invalid native batch write");
    batch->batch->write(offset,
                        {static_cast<const std::byte*>(data), static_cast<std::size_t>(bytes)});
    return {};
  });
}
extern "C" sirius_status sirius_input_publish(sirius_input_handle* input,
                                              sirius_batch_handle** batch,
                                              uint32_t rows,
                                              const sirius_input_vector* columns,
                                              uint32_t count,
                                              sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(input && batch && *batch && (columns || !count) &&
              count <= sirius::embedding::input_columns_limit,
            "invalid native batch publication");
    input->input->publish((*batch)->batch, rows, {columns, count});
    delete *batch;
    *batch = nullptr;
    return {};
  });
}
extern "C" sirius_status sirius_input_finish(sirius_input_handle* input, sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(input, "missing native input");
    input->input->finish();
    return {};
  });
}
extern "C" sirius_status sirius_input_fail(sirius_input_handle* input,
                                           const char* message,
                                           uint32_t bytes,
                                           sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(input && message && bytes < SIRIUS_ERROR_MESSAGE_BYTES, "invalid producer error");
    std::string owned(message, bytes);
    input->input->stop(SIRIUS_EXECUTION_FAILED, owned.c_str());
    input->control->cancel(input->query);
    return {};
  });
}
extern "C" sirius_status sirius_input_get_stats(sirius_input_handle* input,
                                                sirius_input_stats* out,
                                                sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(input && out, "missing native input or stats output");
    require_options(out);
    *out = input->input->inspect();
    return {};
  });
}
extern "C" sirius_status sirius_input_close(sirius_input_handle** input, sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(input, "missing native input handle address");
    if (!*input) return {};
    // Closing a producer is not an implicit successful EOS.
    if (!(*input)->input->finished() && !(*input)->input->outcome().code)
      (*input)->control->cancel((*input)->query);
    --(*input)->query->inputs->handles;
    delete *input;
    *input = nullptr;
    return {};
  });
}
extern "C" sirius_status sirius_batch_release(sirius_batch_handle** batch, sirius_error* error)
{
  return boundary(error, [&]() -> sirius_error {
    require(batch, "missing native batch handle address");
    delete *batch;
    *batch = nullptr;
    return {};
  });
}
