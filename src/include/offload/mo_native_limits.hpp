/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace sirius::offload {

inline constexpr std::size_t max_expanded_native_batch_bytes = 64U * 1024U * 1024U;

[[nodiscard]] inline constexpr std::size_t mo_native_scan_reservation_bytes() noexcept
{
  return max_expanded_native_batch_bytes;
}

}  // namespace sirius::offload
