/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding/control.hpp"
#include "sirius_ffi.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace sirius::embedding {
namespace {
std::atomic<bool> runtime_owned{false};
struct runtime_guard {
  runtime_guard()
  {
    bool expected = false;
    if (!runtime_owned.compare_exchange_strong(expected, true))
      throw failure(SIRIUS_BUSY, "a native GPU runtime already owns this process");
  }
  ~runtime_guard() { runtime_owned.store(false); }
};

class native_backend final : public engine_backend {
 public:
  native_backend(std::string const& config, uint32_t streams)
    : context_(std::make_unique<ffi::Context>(config, streams))
  {
  }

  std::unique_ptr<query_driver> prepare(std::string_view,
                                        std::stop_token,
                                        clock::time_point) override
  {
    // This foundation advertises only ENGINE_CONTROL. Stages 4-6 install the
    // bounded sources and sink before advertising executable native queries.
    // Never fall through to Context::execute_substrait's eager result path.
    throw failure(SIRIUS_UNSUPPORTED, "bounded native query adapters are not installed");
  }

 private:
  runtime_guard guard_;
  std::unique_ptr<ffi::Context> context_;
};
}  // namespace
engine_control::factory native_backend_factory(std::string config_path, uint32_t gpu_streams)
{
  return [path = std::move(config_path), gpu_streams] {
    return std::make_unique<native_backend>(path, gpu_streams);
  };
}
}  // namespace sirius::embedding
