/* Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "embedding/control.hpp"
#include "embedding/input.hpp"
#include "pipeline/gpu_stream_quiescence_error.hpp"
#include "sirius_ffi.hpp"

#include <algorithm>
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
    // Execution requires a checked query contract and registered input bindings.
    // Never fall through to Context::execute_substrait's eager result path.
    throw failure(SIRIUS_UNSUPPORTED, "native queries require a bound query contract");
  }
  std::size_t metadata_capacity_bytes() const noexcept override
  {
    return context_->embedded_metadata_capacity_bytes();
  }
  bool available() const noexcept override { return context_->embedded_runtime_available(); }

  std::unique_ptr<query_driver> prepare_bound(std::string_view plan,
                                              query_state const& query,
                                              input_registry& inputs,
                                              std::stop_token,
                                              clock::time_point) override
  {
    validate_embedded_plan(plan, query);
    for (auto const& binding : query.bindings) {
      auto input = std::find_if(inputs.reads.begin(), inputs.reads.end(), [&](auto const& read) {
        return read->id == binding.binding_id;
      });
      if (binding.source_kind == SIRIUS_READ_TAE) {
        if (input != inputs.reads.end())
          throw failure(SIRIUS_INVALID_ARGUMENT, "TAE reads cannot register an MO producer");
        continue;
      }
      if (input == inputs.reads.end())
        throw failure(SIRIUS_INVALID_ARGUMENT, "MO read binding has no registered producer");
      if ((*input)->schema.size() != binding.columns.size())
        throw failure(SIRIUS_INVALID_ARGUMENT, "MO producer schema does not match read binding");
      for (std::size_t i = 0; i < binding.columns.size(); ++i) {
        auto const& actual   = (*input)->schema[i];
        auto const& expected = binding.columns[i].logical;
        if (actual.oid != expected.oid || actual.width != expected.width ||
            actual.scale != expected.scale || actual.nullable != expected.nullable)
          throw failure(SIRIUS_INVALID_ARGUMENT, "MO producer schema does not match read binding");
      }
    }
    for (auto const& input : inputs.reads) {
      auto binding = std::find_if(query.bindings.begin(), query.bindings.end(), [&](auto const& b) {
        return b.binding_id == input->id && b.source_kind == SIRIUS_READ_MO;
      });
      if (binding == query.bindings.end())
        throw failure(SIRIUS_INVALID_ARGUMENT, "MO producer has no matching read binding");
    }
    std::unique_ptr<ffi::EmbeddedPrepared> prepared;
    try {
      prepared = context_->prepare_embedded(std::string(plan), query, inputs);
    } catch (std::bad_alloc const&) {
      throw;
    } catch (failure const&) {
      throw;
    } catch (pipeline::gpu_stream_quiescence_error const&) {
      throw;
    } catch (std::exception const& e) {
      if (!context_->embedded_runtime_available())
        throw failure(SIRIUS_GPU_UNAVAILABLE, "native preparation poisoned the runtime");
      throw failure(SIRIUS_UNSUPPORTED, e.what());
    }
    class prepared_driver final : public query_driver {
     public:
      explicit prepared_driver(std::unique_ptr<ffi::EmbeddedPrepared> plan) : plan_(std::move(plan))
      {
      }
      void run(std::stop_token stop, clock::time_point deadline) override
      {
        plan_->run(stop, deadline);
      }
      void finish() override
      {
        if (plan_) {
          plan_->finish();
          plan_.reset();
        }
      }
      bool startable() const noexcept override { return true; }

     private:
      std::unique_ptr<ffi::EmbeddedPrepared> plan_;
    };
    try {
      return std::make_unique<prepared_driver>(std::move(prepared));
    } catch (...) {
      prepared.reset();
      if (!context_->embedded_runtime_available())
        throw failure(SIRIUS_GPU_UNAVAILABLE, "native prepared-owner cleanup poisoned the runtime");
      throw;
    }
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
