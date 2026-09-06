/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::pipeline {

/// A CUDA stream could not be synchronized after task failure. The stream and
/// process generation are unsafe for reuse; the paired sidecar must fail stop.
class gpu_stream_quiescence_error final : public std::runtime_error {
 public:
  explicit gpu_stream_quiescence_error(std::string message) : std::runtime_error(std::move(message))
  {
  }
};

}  // namespace sirius::pipeline
