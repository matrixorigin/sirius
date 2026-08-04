/*
 * Copyright 2026, Sirius Contributors.
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

#include <cstdint>
#include <mutex>

namespace sirius {

enum class execution_backend : uint8_t { NONE = 0, SIRIUS_GPU, DUCKDB_CPU };

enum class execution_outcome : uint8_t { PENDING = 0, SUCCEEDED, FAILED, CANCELLED };

struct execution_evidence_snapshot {
  execution_backend planned_backend = execution_backend::NONE;
  execution_backend actual_backend  = execution_backend::NONE;
  execution_outcome outcome         = execution_outcome::PENDING;
  bool fallback                     = false;

  [[nodiscard]] bool is_terminal() const noexcept { return outcome != execution_outcome::PENDING; }
};

/// Query-scoped proof of which engine actually ran an execution.
///
/// The request owner creates this recorder and is the only layer that calls
/// finish(), after any fallback decision. An execution engine calls
/// mark_backend_started() immediately before starting work. Snapshots are
/// immutable values and the first terminal outcome wins.
class execution_evidence final {
 public:
  explicit execution_evidence(execution_backend planned_backend) noexcept;

  execution_evidence(const execution_evidence&)            = delete;
  execution_evidence& operator=(const execution_evidence&) = delete;
  execution_evidence(execution_evidence&&)                 = delete;
  execution_evidence& operator=(execution_evidence&&)      = delete;

  [[nodiscard]] bool mark_backend_started(execution_backend backend) noexcept;
  [[nodiscard]] bool finish(execution_outcome outcome) noexcept;
  [[nodiscard]] execution_evidence_snapshot get_snapshot() const noexcept;

 private:
  mutable std::mutex mutex_;
  execution_evidence_snapshot state_;
};

}  // namespace sirius
