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

#include "execution/sirius_execution_evidence.hpp"

namespace sirius {

namespace {

bool is_valid_backend(execution_backend backend) noexcept
{
  return backend == execution_backend::SIRIUS_GPU || backend == execution_backend::DUCKDB_CPU;
}

bool is_valid_terminal_outcome(execution_outcome outcome) noexcept
{
  return outcome == execution_outcome::SUCCEEDED || outcome == execution_outcome::FAILED ||
         outcome == execution_outcome::CANCELLED;
}

}  // namespace

execution_evidence::execution_evidence(execution_backend planned_backend) noexcept
{
  state_.planned_backend = planned_backend;
}

bool execution_evidence::mark_backend_started(execution_backend backend) noexcept
{
  if (!is_valid_backend(backend)) { return false; }

  std::lock_guard lock(mutex_);
  if (state_.is_terminal()) { return false; }

  if ((state_.actual_backend == execution_backend::NONE &&
       state_.planned_backend != execution_backend::NONE && backend != state_.planned_backend) ||
      (state_.actual_backend != execution_backend::NONE && backend != state_.actual_backend)) {
    state_.fallback = true;
  }
  state_.actual_backend = backend;
  return true;
}

bool execution_evidence::finish(execution_outcome outcome) noexcept
{
  if (!is_valid_terminal_outcome(outcome)) { return false; }

  std::lock_guard lock(mutex_);
  if (state_.is_terminal()) { return false; }
  if (outcome == execution_outcome::SUCCEEDED && state_.actual_backend == execution_backend::NONE) {
    return false;
  }
  state_.outcome = outcome;
  return true;
}

execution_evidence_snapshot execution_evidence::get_snapshot() const noexcept
{
  std::lock_guard lock(mutex_);
  return state_;
}

}  // namespace sirius
