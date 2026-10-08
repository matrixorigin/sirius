/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "numeric/decimal_type.hpp"

#ifdef __CUDACC__
#define SIRIUS_DECIMAL_HD __host__ __device__
#else
#define SIRIUS_DECIMAL_HD
#endif

namespace sirius::mo_decimal {
// Same-scale 64-bit operations need no wide scratch. Unsigned arithmetic makes
// physical overflow checks defined, including signed-minimum negation.
SIRIUS_DECIMAL_HD inline decimal_result evaluate_same_scale64(decimal_op op,
                                                              uint64_t left,
                                                              uint64_t right,
                                                              uint64_t precision_bound,
                                                              bool left_valid,
                                                              bool right_valid,
                                                              bool active)
{
  bool unary = op == decimal_op::cast || op == decimal_op::negate;
  if (!active || !left_valid || (!unary && !right_valid)) return {};
  bool ln = (left >> 63) != 0, rn = (right >> 63) != 0;
  uint64_t value = left;
  auto error = op == decimal_op::cast ? decimal_error::invalid_input : decimal_error::out_of_range;
  if (op >= decimal_op::equal && op <= decimal_op::greater_equal) {
    bool less   = ln != rn ? ln : left < right;
    bool answer = op == decimal_op::equal        ? left == right
                  : op == decimal_op::not_equal  ? left != right
                  : op == decimal_op::less       ? less
                  : op == decimal_op::less_equal ? less || left == right
                  : op == decimal_op::greater    ? !less && left != right
                                                 : !less;
    decimal_result result;
    result.valid          = true;
    result.value.words[0] = answer;
    return result;
  }
  if (op == decimal_op::add) {
    value = left + right;
    if (ln == rn && bool(value >> 63) != ln) return {{}, error, false};
  } else if (op == decimal_op::subtract) {
    value = left - right;
    if (ln != rn && bool(value >> 63) != ln) return {{}, error, false};
  } else if (op == decimal_op::negate) {
    if (left == (uint64_t(1) << 63)) return {{}, error, false};
    value = ~left + 1;
  } else if (op != decimal_op::cast)
    return {{}, decimal_error::invalid_input, false};
  auto negative  = (value >> 63) != 0;
  auto magnitude = negative ? ~value + 1 : value;
  if (magnitude >= precision_bound) return {{}, error, false};
  decimal_result result;
  result.valid          = true;
  result.value.words[0] = value;
  result.value.words[1] = value >> 32;
  for (int i = 2; i < 8; ++i)
    result.value.words[i] = negative ? UINT32_MAX : 0;
  return result;
}

SIRIUS_DECIMAL_HD inline coefficient load_coefficient(uint8_t const* data, uint32_t bytes)
{
  coefficient result;
  auto extension = (data[bytes - 1] & 128) ? UINT32_MAX : 0;
  for (uint32_t i = 0; i < 8; ++i)
    result.words[i] = extension;
  for (uint32_t i = 0; i < bytes / 4; ++i) {
    auto p = data + i * 4;
    result.words[i] =
      uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
  }
  return result;
}
SIRIUS_DECIMAL_HD inline void store_coefficient(coefficient const& value,
                                                uint8_t* data,
                                                uint32_t bytes)
{
  for (uint32_t i = 0; i < bytes; ++i)
    data[i] = value.words[i / 4] >> (i % 4 * 8);
}

namespace decimal_detail {
// Fixed scratch only: no host/device heap, floating point or numeric strings.
struct magnitude {
  uint32_t words[16]{};
  SIRIUS_DECIMAL_HD bool zero() const
  {
    for (auto w : words)
      if (w) return false;
    return true;
  }
};
SIRIUS_DECIMAL_HD inline int compare(magnitude const& a, magnitude const& b)
{
  for (int i = 15; i >= 0; --i) {
    if (a.words[i] < b.words[i]) return -1;
    if (a.words[i] > b.words[i]) return 1;
  }
  return 0;
}
SIRIUS_DECIMAL_HD inline bool add(magnitude& a, magnitude const& b)
{
  uint64_t carry = 0;
  for (int i = 0; i < 16; ++i) {
    auto sum   = uint64_t(a.words[i]) + b.words[i] + carry;
    a.words[i] = sum;
    carry      = sum >> 32;
  }
  return carry != 0;
}
SIRIUS_DECIMAL_HD inline void subtract(magnitude& a, magnitude const& b)
{
  uint64_t borrow = 0;
  for (int i = 0; i < 16; ++i) {
    auto rhs   = uint64_t(b.words[i]) + borrow;
    auto lhs   = uint64_t(a.words[i]);
    a.words[i] = lhs - rhs;
    borrow     = lhs < rhs;
  }
}
SIRIUS_DECIMAL_HD inline bool multiply_small(magnitude& a, uint32_t factor)
{
  uint64_t carry = 0;
  for (int i = 0; i < 16; ++i) {
    auto value = uint64_t(a.words[i]) * factor + carry;
    a.words[i] = value;
    carry      = value >> 32;
  }
  return carry != 0;
}
SIRIUS_DECIMAL_HD inline bool scale_up(magnitude& a, uint32_t digits)
{
  for (uint32_t i = 0; i < digits; ++i)
    if (multiply_small(a, 10)) return false;
  return true;
}
SIRIUS_DECIMAL_HD inline magnitude power10(uint32_t digits)
{
  magnitude result;
  result.words[0] = 1;
  scale_up(result, digits);  // Callers bound digits to at most 152.
  return result;
}
SIRIUS_DECIMAL_HD inline magnitude absolute(coefficient const& c)
{
  magnitude result;
  uint64_t carry = c.negative() ? 1 : 0;
  for (int i = 0; i < 8; ++i) {
    auto word       = c.negative() ? ~c.words[i] : c.words[i];
    auto value      = uint64_t(word) + carry;
    result.words[i] = value;
    carry           = value >> 32;
  }
  return result;
}
SIRIUS_DECIMAL_HD inline magnitude multiply(magnitude const& a, magnitude const& b)
{
  magnitude result;
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      auto value          = uint64_t(a.words[i]) * b.words[j] + result.words[i + j] + carry;
      result.words[i + j] = value;
      carry               = value >> 32;
    }
    result.words[i + 8] = carry;
  }
  return result;
}
SIRIUS_DECIMAL_HD inline bool shift_left(magnitude& a, uint32_t bit)
{
  uint32_t carry = bit;
  for (int i = 0; i < 16; ++i) {
    auto next  = a.words[i] >> 31;
    a.words[i] = (a.words[i] << 1) | carry;
    carry      = next;
  }
  return carry != 0;
}
SIRIUS_DECIMAL_HD inline void divide(magnitude const& numerator,
                                     magnitude const& denominator,
                                     magnitude& quotient,
                                     magnitude& remainder)
{
  quotient  = {};
  remainder = {};
  for (int bit = 511; bit >= 0; --bit) {
    auto carry = shift_left(remainder, (numerator.words[bit / 32] >> (bit % 32)) & 1);
    if (carry || compare(remainder, denominator) >= 0) {
      subtract(remainder, denominator);
      quotient.words[bit / 32] |= uint32_t(1) << (bit % 32);
    }
  }
}
SIRIUS_DECIMAL_HD inline void round(magnitude& quotient,
                                    magnitude const& remainder,
                                    magnitude const& denominator)
{
  magnitude half;
  for (int i = 0; i < 16; ++i)
    half.words[i] = (denominator.words[i] >> 1) | (i == 15 ? 0 : denominator.words[i + 1] << 31);
  magnitude one;
  one.words[0] = 1;
  if (denominator.words[0] & 1) add(half, one);
  if (compare(remainder, half) >= 0) add(quotient, one);
}
SIRIUS_DECIMAL_HD inline bool rescale(magnitude& value, int from, int to, bool rounding)
{
  if (to >= from) return scale_up(value, to - from);
  auto denominator = power10(from - to);
  magnitude quotient, remainder;
  divide(value, denominator, quotient, remainder);
  if (rounding) round(quotient, remainder, denominator);
  value = quotient;
  return true;
}
SIRIUS_DECIMAL_HD inline decimal_result finish(magnitude const& value,
                                               bool negative,
                                               decimal_type type,
                                               decimal_error error,
                                               bool precision_check = true)
{
  magnitude physical;
  physical.words[(type.bits - 1) / 32] = uint32_t(1) << ((type.bits - 1) % 32);
  auto comparison                      = compare(value, physical);
  if (comparison > 0 || (comparison == 0 && !negative) ||
      (precision_check && compare(value, power10(type.precision)) >= 0))
    return {{}, error, false};
  decimal_result result;
  result.valid   = true;
  uint64_t carry = negative && !value.zero() ? 1 : 0;
  for (int i = 0; i < 8; ++i) {
    auto word             = negative && !value.zero() ? ~value.words[i] : value.words[i];
    auto v                = uint64_t(word) + carry;
    result.value.words[i] = v;
    carry                 = v >> 32;
  }
  return result;
}
}  // namespace decimal_detail

// Descriptor validation is repeated by the host launch wrapper. This guard
// also keeps direct host/device consumers total for malformed descriptors.
SIRIUS_DECIMAL_HD inline decimal_result evaluate_decimal(decimal_op op,
                                                         coefficient const& left,
                                                         decimal_type left_type,
                                                         bool left_valid,
                                                         coefficient const& right,
                                                         decimal_type right_type,
                                                         bool right_valid,
                                                         decimal_type output_type,
                                                         bool active = true)
{
  using namespace decimal_detail;
  auto error = op == decimal_op::cast ? decimal_error::invalid_input : decimal_error::out_of_range;
  if (!left_type.valid() || !right_type.valid() || !output_type.valid() ||
      static_cast<uint8_t>(op) > static_cast<uint8_t>(decimal_op::greater_equal))
    return {{}, decimal_error::invalid_input, false};
  bool unary = op == decimal_op::cast || op == decimal_op::negate;
  if (!active || !left_valid || (!unary && !right_valid)) return {};
  if (left_type.bits == 64 && right_type.bits == 64 && output_type.bits == 64 &&
      left_type.scale == right_type.scale && left_type.scale == output_type.scale &&
      (op == decimal_op::add || op == decimal_op::subtract || op == decimal_op::negate ||
       op == decimal_op::cast || op >= decimal_op::equal)) {
    uint64_t bound = 1;
    for (int i = 0; i < output_type.precision; ++i)
      bound *= 10;
    return evaluate_same_scale64(op,
                                 uint64_t(left.words[0]) | (uint64_t(left.words[1]) << 32),
                                 uint64_t(right.words[0]) | (uint64_t(right.words[1]) << 32),
                                 bound,
                                 left_valid,
                                 right_valid,
                                 active);
  }
  auto a = absolute(left), b = absolute(right);
  bool negative = left.negative();
  int scale     = left_type.scale;
  switch (op) {
    case decimal_op::cast: break;
    case decimal_op::negate: negative = !negative; break;
    case decimal_op::multiply:
      a = multiply(a, b);
      scale += right_type.scale;
      negative = left.negative() != right.negative();
      break;
    case decimal_op::divide:
    case decimal_op::integer_divide: {
      if (b.zero()) return {};
      auto target_scale = op == decimal_op::integer_divide ? 0 : output_type.scale;
      int exponent      = target_scale + int(right_type.scale) - int(left_type.scale);
      if (exponent >= 0) {
        // If this overflows 512 bits, division by any 256-bit coefficient
        // cannot produce a representable signed 256-bit result.
        if (!scale_up(a, exponent)) return {{}, error, false};
      } else if (!scale_up(b, -exponent))
        return {{}, error, false};
      magnitude quotient, remainder;
      divide(a, b, quotient, remainder);
      if (op == decimal_op::divide) round(quotient, remainder, b);
      negative = left.negative() != right.negative();
      if (op == decimal_op::integer_divide)
        return finish(quotient, negative, {64, 18, 0}, error, false);
      return finish(quotient, negative, output_type, error);
    }
    default: {
      scale = left_type.scale > right_type.scale ? left_type.scale : right_type.scale;
      if (!scale_up(a, scale - left_type.scale) || !scale_up(b, scale - right_type.scale))
        return {{}, error, false};
      if (op == decimal_op::modulo) {
        if (b.zero()) return {};
        magnitude quotient, remainder;
        divide(a, b, quotient, remainder);
        a = remainder;
        break;
      }
      auto rhs_negative = right.negative() != (op == decimal_op::subtract);
      auto cmp          = compare(a, b);
      if (op >= decimal_op::equal) {
        if (a.zero() && b.zero())
          cmp = 0;
        else if (left.negative() != right.negative())
          cmp = left.negative() ? -1 : 1;
        else if (left.negative())
          cmp = -cmp;
        bool answer = op == decimal_op::equal        ? cmp == 0
                      : op == decimal_op::not_equal  ? cmp != 0
                      : op == decimal_op::less       ? cmp < 0
                      : op == decimal_op::less_equal ? cmp <= 0
                      : op == decimal_op::greater    ? cmp > 0
                                                     : cmp >= 0;
        decimal_result result;
        result.valid          = true;
        result.value.words[0] = answer;
        return result;
      }
      if (op != decimal_op::add && op != decimal_op::subtract)
        return {{}, decimal_error::invalid_input, false};
      if (left.negative() == rhs_negative) {
        if (add(a, b)) return {{}, error, false};
      } else if (cmp >= 0)
        subtract(a, b);
      else {
        subtract(b, a);
        a        = b;
        negative = rhs_negative;
      }
      break;
    }
  }
  if (!rescale(a, scale, output_type.scale, op != decimal_op::modulo)) return {{}, error, false};
  return finish(a, negative, output_type, error);
}
}  // namespace sirius::mo_decimal

#undef SIRIUS_DECIMAL_HD
