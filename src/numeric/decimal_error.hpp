/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "numeric/decimal_type.hpp"

#include <stdexcept>
namespace sirius::mo_decimal {
class numeric_error final : public std::runtime_error {
 public:
  explicit numeric_error(decimal_error code)
    : std::runtime_error(code == decimal_error::invalid_input
                           ? "MO exact-decimal checked cast failed"
                           : "MO exact-decimal arithmetic is out of range"),
      code_(code)
  {
  }
  decimal_error code() const noexcept { return code_; }

 private:
  decimal_error code_;
};
}  // namespace sirius::mo_decimal
