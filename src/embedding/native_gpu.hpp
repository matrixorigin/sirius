/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "embedding/input.hpp"
#include "op/scan/gpu_ingestible.hpp"

namespace sirius::embedding {
// Admission reserves a complete read window before any producer allocates.
std::shared_ptr<input_pool> make_native_input_pool(cucascade::memory::memory_space& host,
                                                   std::size_t capacity = input_window);
void activate_native_inputs(input_registry& inputs, cucascade::memory::memory_space& host);
std::unique_ptr<cudf::table> convert_native_input(input_unit const& unit,
                                                  std::span<const sirius_input_column> schema,
                                                  cucascade::memory::memory_space const& gpu,
                                                  rmm::cuda_stream_view stream);
std::shared_ptr<op::scan::gpu_ingestible> make_native_ingestible(
  std::shared_ptr<native_input> input);
}  // namespace sirius::embedding
