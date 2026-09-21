/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "embedding/buffer_budget.hpp"

#include <cstdint>
#include <memory>

namespace sirius::embedding {

// Assigned under the terminal pipeline's input-claim lock. The credit outlives
// compute when a bounded publication cursor retains this ticket.
struct terminal_ticket {
  std::uint64_t sequence{0};
  std::shared_ptr<void> credit;
};

class terminal_admission {
 public:
  virtual ~terminal_admission() = default;
  // Null means temporarily full or sealed; never waits for a result consumer.
  virtual std::shared_ptr<terminal_ticket> try_acquire() = 0;
  // May have more than one final pipeline. Retain subscribers weakly and notify
  // without holding the admission mutex. Their close precedes plan teardown.
  virtual void subscribe(std::shared_ptr<capacity_waker> wake) = 0;
};

}  // namespace sirius::embedding
