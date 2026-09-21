/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <stdexcept>
namespace sirius::embedding {
class execution_interrupted : public std::runtime_error {
 public:
  explicit execution_interrupted(bool deadline)
    : std::runtime_error(deadline ? "native execution deadline expired"
                                  : "native execution cancelled"),
      deadline_expired(deadline)
  {
  }
  bool const deadline_expired;
};
}  // namespace sirius::embedding
