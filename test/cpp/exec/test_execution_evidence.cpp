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

#include <catch.hpp>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

using sirius::execution_backend;
using sirius::execution_evidence;
using sirius::execution_outcome;

TEST_CASE("execution evidence records the backend that produced the result", "[execution_evidence]")
{
  execution_evidence evidence(execution_backend::SIRIUS_GPU);

  REQUIRE(evidence.mark_backend_started(execution_backend::SIRIUS_GPU));
  REQUIRE(evidence.finish(execution_outcome::SUCCEEDED));

  auto snapshot = evidence.get_snapshot();
  REQUIRE(snapshot.planned_backend == execution_backend::SIRIUS_GPU);
  REQUIRE(snapshot.actual_backend == execution_backend::SIRIUS_GPU);
  REQUIRE(snapshot.outcome == execution_outcome::SUCCEEDED);
  REQUIRE_FALSE(snapshot.fallback);
}

TEST_CASE("execution evidence records CPU fallback after a GPU attempt", "[execution_evidence]")
{
  execution_evidence evidence(execution_backend::SIRIUS_GPU);

  REQUIRE(evidence.mark_backend_started(execution_backend::SIRIUS_GPU));
  REQUIRE(evidence.mark_backend_started(execution_backend::DUCKDB_CPU));
  REQUIRE(evidence.finish(execution_outcome::SUCCEEDED));

  auto snapshot = evidence.get_snapshot();
  REQUIRE(snapshot.actual_backend == execution_backend::DUCKDB_CPU);
  REQUIRE(snapshot.outcome == execution_outcome::SUCCEEDED);
  REQUIRE(snapshot.fallback);
}

TEST_CASE("execution evidence accepts failure and cancellation before execution",
          "[execution_evidence]")
{
  SECTION("validation failure")
  {
    execution_evidence evidence(execution_backend::SIRIUS_GPU);
    REQUIRE(evidence.finish(execution_outcome::FAILED));
    auto snapshot = evidence.get_snapshot();
    REQUIRE(snapshot.actual_backend == execution_backend::NONE);
    REQUIRE(snapshot.outcome == execution_outcome::FAILED);
  }

  SECTION("cancellation")
  {
    execution_evidence evidence(execution_backend::SIRIUS_GPU);
    REQUIRE(evidence.finish(execution_outcome::CANCELLED));
    REQUIRE(evidence.get_snapshot().outcome == execution_outcome::CANCELLED);
  }
}

TEST_CASE("execution evidence rejects impossible transitions", "[execution_evidence]")
{
  execution_evidence evidence(execution_backend::SIRIUS_GPU);

  REQUIRE_FALSE(evidence.finish(execution_outcome::SUCCEEDED));
  REQUIRE_FALSE(evidence.finish(execution_outcome::PENDING));
  REQUIRE_FALSE(evidence.finish(static_cast<execution_outcome>(255)));
  REQUIRE_FALSE(evidence.mark_backend_started(execution_backend::NONE));
  REQUIRE_FALSE(evidence.mark_backend_started(static_cast<execution_backend>(255)));
  REQUIRE_FALSE(evidence.get_snapshot().is_terminal());

  REQUIRE(evidence.mark_backend_started(execution_backend::SIRIUS_GPU));
  REQUIRE(evidence.finish(execution_outcome::SUCCEEDED));
}

TEST_CASE("execution evidence first terminal transition wins", "[execution_evidence]")
{
  auto evidence = std::make_shared<execution_evidence>(execution_backend::SIRIUS_GPU);
  REQUIRE(evidence->mark_backend_started(execution_backend::SIRIUS_GPU));

  std::atomic<int> winners{0};
  std::vector<std::thread> threads;
  for (auto outcome :
       {execution_outcome::SUCCEEDED, execution_outcome::FAILED, execution_outcome::CANCELLED}) {
    threads.emplace_back([evidence, outcome, &winners] {
      if (evidence->finish(outcome)) { winners.fetch_add(1); }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(winners.load() == 1);
  REQUIRE(evidence->get_snapshot().is_terminal());
  REQUIRE_FALSE(evidence->mark_backend_started(execution_backend::DUCKDB_CPU));
  REQUIRE_FALSE(evidence->finish(execution_outcome::FAILED));
  REQUIRE_FALSE(evidence->finish(execution_outcome::PENDING));
}

TEST_CASE("execution evidence is isolated per query", "[execution_evidence]")
{
  execution_evidence gpu(execution_backend::SIRIUS_GPU);
  execution_evidence fallback(execution_backend::SIRIUS_GPU);

  REQUIRE(gpu.mark_backend_started(execution_backend::SIRIUS_GPU));
  REQUIRE(fallback.mark_backend_started(execution_backend::DUCKDB_CPU));
  REQUIRE(gpu.finish(execution_outcome::SUCCEEDED));
  REQUIRE(fallback.finish(execution_outcome::FAILED));

  auto gpu_snapshot      = gpu.get_snapshot();
  auto fallback_snapshot = fallback.get_snapshot();
  REQUIRE(gpu_snapshot.actual_backend == execution_backend::SIRIUS_GPU);
  REQUIRE_FALSE(gpu_snapshot.fallback);
  REQUIRE(fallback_snapshot.actual_backend == execution_backend::DUCKDB_CPU);
  REQUIRE(fallback_snapshot.fallback);
  REQUIRE(fallback_snapshot.outcome == execution_outcome::FAILED);
}
