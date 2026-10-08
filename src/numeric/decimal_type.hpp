/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define SIRIUS_DECIMAL_TYPE_HD __host__ __device__
#else
#define SIRIUS_DECIMAL_TYPE_HD
#endif
namespace sirius::mo_decimal {
// MO physical width is independent of SQL precision. This descriptor does not
// change ordinary DuckDB/cuDF decimal type inference.
struct decimal_type {
  uint16_t bits;
  uint8_t precision, scale;
  SIRIUS_DECIMAL_TYPE_HD bool valid() const
  {
    auto limit = bits == 64 ? 18 : bits == 128 ? 38 : bits == 256 ? 76 : 0;
    return precision > 0 && precision <= limit && scale <= precision;
  }
  SIRIUS_DECIMAL_TYPE_HD uint32_t bytes() const { return bits / 8; }
};

struct coefficient {
  uint32_t words[8]{};  // Little-endian, signed two's complement, sign extended.
  SIRIUS_DECIMAL_TYPE_HD bool negative() const { return (words[7] >> 31) != 0; }
};
enum class decimal_error : uint8_t { none, out_of_range, invalid_input };
enum class decimal_op : uint8_t {
  add,
  subtract,
  multiply,
  divide,
  integer_divide,
  modulo,
  negate,
  cast,
  equal,
  not_equal,
  less,
  less_equal,
  greater,
  greater_equal
};
struct decimal_result {
  coefficient value{};
  decimal_error error{decimal_error::none};
  bool valid{false};
};

}  // namespace sirius::mo_decimal
#undef SIRIUS_DECIMAL_TYPE_HD
