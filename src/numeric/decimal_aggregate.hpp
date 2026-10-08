/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "numeric/exact_decimal.hpp"

#ifdef __CUDACC__
#define SIRIUS_AGG_HD __host__ __device__
#else
#define SIRIUS_AGG_HD
#endif
namespace sirius::mo_decimal {
enum class aggregate_op : uint8_t { sum, avg, min, max };

// Raw physical state: its coefficient deliberately has no declared precision.
// All partials retain the input scale, and count zero is the NULL identity.
struct aggregate_state {
  coefficient value{};
  int64_t count{};
  decimal_error error{decimal_error::none};
};
SIRIUS_AGG_HD inline bool fits_physical(coefficient const& value, uint16_t bits)
{
  if (bits != 128 && bits != 256) return false;
  auto words     = bits / 32;
  auto extension = value.words[words - 1] >> 31 ? UINT32_MAX : 0;
  for (int i = words; i < 8; ++i)
    if (value.words[i] != extension) return false;
  return true;
}
SIRIUS_AGG_HD inline int compare_coefficient(coefficient const& a, coefficient const& b)
{
  if (a.negative() != b.negative()) return a.negative() ? -1 : 1;
  for (int i = 7; i >= 0; --i) {
    if (a.words[i] < b.words[i]) return -1;
    if (a.words[i] > b.words[i]) return 1;
  }
  return 0;
}
SIRIUS_AGG_HD inline aggregate_state merge_aggregate(aggregate_op op,
                                                     aggregate_state left,
                                                     aggregate_state const& right,
                                                     uint16_t bits)
{
  if (left.error != decimal_error::none) return left;
  if (right.error != decimal_error::none) return right;
  if (op > aggregate_op::max || (bits != 128 && bits != 256) || left.count < 0 || right.count < 0 ||
      right.count > INT64_MAX - left.count)
    return {{}, 0, decimal_error::invalid_input};
  if (!right.count) return left;
  if (!left.count) {
    if (!fits_physical(right.value, bits)) return {{}, 0, decimal_error::invalid_input};
    return right;
  }
  if (op == aggregate_op::min || op == aggregate_op::max) {
    auto cmp = compare_coefficient(left.value, right.value);
    if ((op == aggregate_op::min && cmp > 0) || (op == aggregate_op::max && cmp < 0))
      left.value = right.value;
  } else {
    auto negative  = left.value.negative();
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
      auto sum            = uint64_t(left.value.words[i]) + right.value.words[i] + carry;
      left.value.words[i] = sum;
      carry               = sum >> 32;
    }
    if ((negative == right.value.negative() && left.value.negative() != negative) ||
        !fits_physical(left.value, bits))
      return {{}, 0, decimal_error::invalid_input};
  }
  left.count += right.count;
  return left;
}
SIRIUS_AGG_HD inline aggregate_state accumulate_aggregate(
  aggregate_op op, aggregate_state state, coefficient value, bool valid, uint16_t bits)
{
  return valid ? merge_aggregate(op, state, {value, 1, decimal_error::none}, bits) : state;
}
SIRIUS_AGG_HD inline decimal_result finalize_aggregate(aggregate_op op,
                                                       aggregate_state const& state,
                                                       decimal_type input,
                                                       decimal_type output)
{
  if (state.error != decimal_error::none) return {{}, state.error, false};
  if (!input.valid() || !output.valid() || op > aggregate_op::max || state.count < 0)
    return {{}, decimal_error::invalid_input, false};
  if (!state.count) return {};
  if (op == aggregate_op::min || op == aggregate_op::max)
    return {state.value, decimal_error::none, true};
  if (op == aggregate_op::sum) {
    if (input.scale != output.scale) return {{}, decimal_error::invalid_input, false};
    return decimal_detail::finish(decimal_detail::absolute(state.value),
                                  state.value.negative(),
                                  output,
                                  decimal_error::invalid_input);
  }
  coefficient count;
  count.words[0] = static_cast<uint64_t>(state.count);
  count.words[1] = static_cast<uint64_t>(state.count) >> 32;
  // The scalar division core never validates the input coefficient against
  // precision. This descriptor expresses only the raw physical state/scale.
  auto result = evaluate_decimal(decimal_op::divide,
                                 state.value,
                                 {256, 76, input.scale},
                                 true,
                                 count,
                                 {64, 18, 0},
                                 true,
                                 output);
  if (result.error != decimal_error::none) result.error = decimal_error::invalid_input;
  return result;
}

struct normalized_decimal_key {
  coefficient value{};
  uint8_t scale{};
};
SIRIUS_AGG_HD inline normalized_decimal_key normalize_key(coefficient value, uint8_t scale)
{
  auto magnitude = decimal_detail::absolute(value);
  if (magnitude.zero()) return {};
  while (scale) {
    auto quotient      = magnitude;
    uint64_t remainder = 0;
    for (int i = 7; i >= 0; --i) {
      auto word         = (remainder << 32) | magnitude.words[i];
      quotient.words[i] = word / 10;
      remainder         = word % 10;
    }
    if (remainder) break;
    magnitude = quotient;
    --scale;
  }
  return {decimal_detail::finish(
            magnitude, value.negative(), {256, 76, 0}, decimal_error::invalid_input, false)
            .value,
          scale};
}
}  // namespace sirius::mo_decimal
#undef SIRIUS_AGG_HD
