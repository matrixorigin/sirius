/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "offload/mo_native_batch.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_table_scan.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace sirius::op {

class sirius_physical_gpu_mo_scan final : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GPU_MO_SCAN;

  explicit sirius_physical_gpu_mo_scan(sirius_physical_table_scan* table_scan);

  std::optional<task_creation_hint> get_next_task_hint() override
  {
    if (exhausted.load(std::memory_order_acquire)) { return std::nullopt; }
    // This source permits exactly one blocking next_batch() task. Reporting
    // WAITING with itself as producer would make task selection recurse back
    // into this same operator while that task is active. Publication releases
    // the claim without self-scheduling; only downstream demand may reach this
    // source again and admit the next task.
    if (task_active.load(std::memory_order_acquire)) { return std::nullopt; }
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  bool is_source() const override { return true; }

  std::shared_ptr<offload::mo_native_batch_source> source;
  std::vector<std::size_t> source_column_ids;
  std::atomic<bool> task_active{false};
  std::atomic<bool> exhausted{false};
};

}  // namespace sirius::op
