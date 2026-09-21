/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "exec/completion_controller.hpp"
#include "exec/try.hpp"
#include "io/io_context.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "scan_manager/split_connector.hpp"

#include <functional>
#include <memory>
#include <thread>

namespace sirius::op {
class operator_data;
}  // namespace sirius::op

namespace sirius::scan_manager {

/**
 * @brief Driver of splits for a scan operator.
 *
 * Borrows a @c gpu_ingestible and delegates the per-format work to it.
 * Exposes two thread-safe primitives:
 *   - @ref has_more_splits — snapshot check for remaining work.
 *   - @ref next_split_provider — atomically claim the next batch and return a
 *     callable that processes it.
 *
 * @ref run() launches one producer thread per scan. That thread claims and
 * processes one metadata unit at a time, then sends the result through the
 * bounded callback. The callback may apply back-pressure without occupying a
 * shared dispatcher worker. Completion closes the connector, and producer
 * failures are forwarded through the same callback so consumers see them via
 * @ref split_connector::get_next_split.
 *
 * Pinned scans are served by @c cached_databatch_provider instead.
 */
class split_provider {
 public:
  using value_type      = exec::try_t<std::unique_ptr<op::scan::scan_info>>;
  using push_callback_t = std::function<void(value_type&&)>;

  /// Concrete construction path — borrows the given ingestible non-owningly.
  /// The ingestible must outlive the provider; in practice the operator owns
  /// the ingestible (single shared_ptr) and the provider is created after
  /// install and torn down by scan_manager reset() before the operator goes
  /// away. Callers that need a shared_ptr can promote via
  /// `provider.get_ingestible().shared_from_this()` (enabled by
  /// @c gpu_ingestible inheriting @c std::enable_shared_from_this).
  explicit split_provider(op::scan::gpu_ingestible& ingestible, io::ioctx_resolver resolve);

  virtual ~split_provider() = default;

  /**
   * @brief Start the provider's per-scan producer thread.
   *
   * The producer claims and executes one metadata task at a time. Its bounded
   * @p on_split callback may block until downstream capacity is available.
   * This intentionally keeps blocking reads and queue waits off the shared
   * scan-manager dispatcher. The connector is closed after producer
   * completion; exceptions are delivered through @p on_split.
   *
   * @note The producer is joined when the provider is destroyed. Call
   *       @ref request_stop before teardown to interrupt a blocked metadata
   *       read or callback.
   */
  void run(const push_callback_t& on_split);
  void request_stop() noexcept
  {
    if (_producer.joinable()) _producer.request_stop();
  }

  /**
   * @brief Snapshot check for remaining work. Thread-safe.
   *
   * Delegates to the borrowed @c gpu_ingestible.
   */
  [[nodiscard]] virtual bool has_more_splits() const;

  /**
   * @brief Atomically claim the next batch and return a callable that
   *        produces its splits. Thread-safe.
   *
   * Delegates to the borrowed @c gpu_ingestible. Returns
   * a null @c std::function when no batch is available — callers that
   * already checked @ref has_more_splits() will normally not hit this
   * path, but the null fallback keeps the contract simple under concurrent
   * observers.
   */
  virtual std::function<std::unique_ptr<op::scan::scan_info>()> next_split_provider();

  /// Accessor for the borrowed ingestible. Callers that need a
  /// @c shared_ptr<gpu_ingestible> can promote via
  /// `provider.get_ingestible().shared_from_this()`.
  [[nodiscard]] op::scan::gpu_ingestible& get_ingestible() const noexcept { return *_ingestible; }

 private:
  /// Non-owning pointer to the borrowed ingestible. The operator owns the
  /// lifetime; the provider is always destroyed first via
  /// @c sirius_scan_manager::reset.
  op::scan::gpu_ingestible* _ingestible{nullptr};

  /// Resolves each file's ioctx by path (s3:// -> rest, local -> uring/kvikio),
  /// forwarded into the ingestible so a mixed-scheme scan routes per file.
  io::ioctx_resolver _resolve;

  exec::completion_token _completion_token;
  // A bounded-output producer can wait here without consuming a shared
  // dispatcher worker or preventing query setup from completing.
  std::jthread _producer;
};

}  // namespace sirius::scan_manager
