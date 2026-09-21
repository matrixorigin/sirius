/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>

namespace sirius::embedding {
using tae_read_at = std::function<std::size_t(std::uint64_t, std::size_t, std::uint8_t*)>;

// Read and strip CRC framing IN PLACE. The supplied charged allocation is the
// complete transport scratch: no raw/stripped side buffers are allocated. A
// transport slice may bisect an LZ4 chunk; callers assemble it on the device.
inline std::size_t read_tae_slice(tae_read_at const& read,
                                  std::uint64_t physical_size,
                                  bool crc,
                                  std::uint64_t logical_offset,
                                  std::size_t remaining,
                                  std::span<std::uint8_t> staging)
{
  if (!remaining) return 0;
  if (staging.empty()) throw std::invalid_argument("empty TAE staging slice");
  if (!crc) {
    auto const bytes = std::min(remaining, staging.size());
    if (logical_offset > physical_size || bytes > physical_size - logical_offset ||
        read(logical_offset, bytes, staging.data()) != bytes)
      throw std::runtime_error("short TAE payload read");
    return bytes;
  }
  constexpr std::uint64_t block = 2048, content = 2044, checksum = 4;
  if (staging.size() < block) throw std::invalid_argument("TAE CRC slice is smaller than a block");
  auto const first = logical_offset / content;
  if (first > physical_size / block) throw std::runtime_error("TAE CRC offset outside object");
  auto const local = static_cast<std::size_t>(logical_offset % content);
  if (first * block > physical_size || checksum + local > physical_size - first * block)
    throw std::runtime_error("TAE CRC content offset outside object");
  auto const begin = first * block + checksum + local;
  // Leave room for every intervening checksum. Start/end at the exact logical
  // range, not the surrounding CRC block: metadata reads must not pull payload
  // bytes that happen to share their first/last physical block.
  auto const logical_capacity = staging.size() - (staging.size() / content + 2) * checksum;
  auto const wanted           = std::min<std::uint64_t>(remaining, logical_capacity);
  if (wanted > physical_size - begin) throw std::runtime_error("TAE CRC content is truncated");
  auto const crossed   = (local + wanted - 1) / content;
  auto const raw_bytes = static_cast<std::size_t>(wanted + crossed * checksum);
  if (raw_bytes > physical_size - begin) throw std::runtime_error("TAE CRC content is truncated");
  if (read(begin, raw_bytes, staging.data()) != raw_bytes)
    throw std::runtime_error("short TAE CRC payload read");
  std::size_t stripped = 0;
  for (std::size_t offset = 0; offset < raw_bytes;) {
    auto const physical_local = (begin + offset) % block;
    if (physical_local < checksum) {
      offset += std::min<std::size_t>(checksum - physical_local, raw_bytes - offset);
      continue;
    }
    auto const bytes = std::min<std::size_t>(block - physical_local, raw_bytes - offset);
    std::memmove(staging.data() + stripped, staging.data() + offset, bytes);
    stripped += bytes;
    offset += bytes;
  }
  if (wanted != stripped) throw std::runtime_error("TAE CRC content is truncated");
  return wanted;
}
}  // namespace sirius::embedding
