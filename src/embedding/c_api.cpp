/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding/control.hpp"
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
