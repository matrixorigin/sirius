/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tae/tae_format.hpp"

#include <duckdb/function/function.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace sirius::offload {

struct mo_native_column_view {
  std::uint8_t vector_class = 0;
  tae::MOType type{};
  std::uint32_t logical_rows = 0;
  std::string_view encoded;
  std::string_view data;
  std::string_view area;
  std::string_view null_words;
  std::uint64_t null_count = 0;
  bool sorted              = false;

  [[nodiscard]] bool is_null(std::uint64_t row) const noexcept
  {
    if (null_count == 0 || null_words.empty()) { return false; }
    const auto offset = static_cast<std::size_t>(row / 64U) * sizeof(std::uint64_t);
    if (offset + sizeof(std::uint64_t) > null_words.size()) { return false; }
    std::uint64_t word = 0;
    for (unsigned i = 0; i < sizeof(std::uint64_t); ++i) {
      word |= static_cast<std::uint64_t>(static_cast<unsigned char>(null_words[offset + i]))
              << (i * 8U);
    }
    return (word & (std::uint64_t{1} << (row % 64U))) != 0;
  }
};

class mo_native_batch {
 public:
  virtual ~mo_native_batch()                                         = default;
  [[nodiscard]] virtual std::uint64_t sequence() const noexcept      = 0;
  [[nodiscard]] virtual std::uint64_t rows() const noexcept          = 0;
  [[nodiscard]] virtual std::uint64_t payload_bytes() const noexcept = 0;
  [[nodiscard]] virtual const std::vector<mo_native_column_view>& columns() const noexcept = 0;
};

enum class mo_native_batch_source_status { BATCH, END_OF_STREAM, NOT_NEEDED };

struct mo_native_batch_source_result {
  mo_native_batch_source_status status = mo_native_batch_source_status::END_OF_STREAM;
  std::shared_ptr<mo_native_batch> batch;
};

class mo_native_batch_source {
 public:
  virtual ~mo_native_batch_source()                  = default;
  virtual mo_native_batch_source_result next_batch() = 0;
  virtual bool mark_consumed(std::uint64_t sequence) = 0;
};

// DuckDB binds this marker to mo_stream_scan. Sirius recognizes the common
// bind-data type and replaces the CPU table-function source with its native
// MO-vector GPU scan. Copying retains the query-owned source lifetime.
class mo_native_scan_bind_data final : public duckdb::FunctionData {
 public:
  explicit mo_native_scan_bind_data(std::shared_ptr<mo_native_batch_source> source_p)
    : source(std::move(source_p))
  {
  }

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    return duckdb::make_uniq<mo_native_scan_bind_data>(source);
  }

  bool Equals(const duckdb::FunctionData& other) const override
  {
    auto const* rhs = dynamic_cast<const mo_native_scan_bind_data*>(&other);
    return rhs != nullptr && rhs->source.get() == source.get();
  }

  std::shared_ptr<mo_native_batch_source> source;
};

}  // namespace sirius::offload
