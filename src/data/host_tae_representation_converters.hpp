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

// cucascade
#include <cucascade/data/representation_converter.hpp>
#include <data/host_tae_representation.hpp>

namespace sirius {

/**
 * @brief Register converters for host_tae_representation → gpu_table_representation.
 */
void register_tae_converters(cucascade::representation_converter_registry& registry);

// Checked conservative peak for descriptor-only embedded TAE tasks. Includes
// compressed mirror, decompression/scratch, decode output and filter copies.
std::size_t tae_decode_reservation_floor(
  std::vector<host_tae_representation::column_chunk_info> const& chunks);

}  // namespace sirius
