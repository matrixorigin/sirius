/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/decimal_aggregate.hpp"

#include <catch.hpp>

#include <algorithm>
#include <array>

using namespace sirius::mo_decimal;
namespace {
coefficient integer(int64_t value)
{
  return load_coefficient(reinterpret_cast<uint8_t const*>(&value), 8);
}
void equal(coefficient const& observed, coefficient const& expected)
{
  for (int i = 0; i < 8; ++i)
    CHECK(observed.words[i] == expected.words[i]);
}
}  // namespace
TEST_CASE("MO aggregate states retain physical range until final publication",
          "[decimal_aggregate]")
{
  auto large = decimal_detail::power10(65);
  decimal_detail::magnitude one;
  one.words[0] = 1;
  decimal_detail::subtract(large, one);
  auto positive =
    decimal_detail::finish(large, false, {256, 65, 0}, decimal_error::invalid_input).value;
  auto negative =
    decimal_detail::finish(large, true, {256, 65, 0}, decimal_error::invalid_input).value;
  std::array<int, 4> order{0, 1, 2, 3};
  std::array<coefficient, 4> values{positive, positive, negative, negative};
  do {
    aggregate_state state;
    for (auto index : order)
      state = accumulate_aggregate(aggregate_op::sum, state, values[index], true, 256);
    REQUIRE(state.error == decimal_error::none);
    auto result = finalize_aggregate(aggregate_op::sum, state, {256, 65, 0}, {256, 65, 0});
    REQUIRE(result.valid);
    equal(result.value, {});
  } while (std::next_permutation(order.begin(), order.end()));
  auto state = accumulate_aggregate(aggregate_op::sum, {}, positive, true, 256);
  state      = accumulate_aggregate(aggregate_op::sum, state, integer(1), true, 256);
  CHECK(state.error == decimal_error::none);
  CHECK(finalize_aggregate(aggregate_op::sum, state, {256, 65, 0}, {256, 65, 0}).error ==
        decimal_error::invalid_input);
  coefficient physical_max;
  for (int i = 0; i < 8; ++i)
    physical_max.words[i] = UINT32_MAX;
  physical_max.words[7] >>= 1;
  state = accumulate_aggregate(aggregate_op::sum, {}, physical_max, true, 256);
  state = accumulate_aggregate(aggregate_op::sum, state, integer(1), true, 256);
  CHECK(state.error == decimal_error::invalid_input);
  coefficient physical128;
  for (int i = 0; i < 4; ++i)
    physical128.words[i] = UINT32_MAX;
  physical128.words[3] >>= 1;
  state = accumulate_aggregate(aggregate_op::sum, {}, physical128, true, 128);
  state = accumulate_aggregate(aggregate_op::sum, state, integer(1), true, 128);
  CHECK(state.error == decimal_error::invalid_input);
}
TEST_CASE("MO aggregate empty identities and signed AVG round only once", "[decimal_aggregate]")
{
  for (auto op : {aggregate_op::sum, aggregate_op::avg, aggregate_op::min, aggregate_op::max}) {
    auto empty = accumulate_aggregate(op, {}, integer(999), false, 128);
    CHECK(empty.count == 0);
    CHECK_FALSE(finalize_aggregate(op, empty, {64, 9, 2}, {128, 19, 6}).valid);
  }
  for (auto sign : {-1, 1}) {
    auto left   = accumulate_aggregate(aggregate_op::avg, {}, integer(sign * 124), true, 128);
    auto right  = accumulate_aggregate(aggregate_op::avg, {}, integer(sign * 126), true, 128);
    auto state  = merge_aggregate(aggregate_op::avg, left, right, 128);
    auto result = finalize_aggregate(aggregate_op::avg, state, {64, 9, 2}, {128, 19, 1});
    REQUIRE(result.valid);
    equal(result.value, integer(sign * 13));
  }
  auto min = accumulate_aggregate(aggregate_op::min, {}, integer(7), true, 128);
  min      = accumulate_aggregate(aggregate_op::min, min, integer(-9), true, 128);
  equal(finalize_aggregate(aggregate_op::min, min, {64, 9, 0}, {64, 9, 0}).value, integer(-9));
  auto max = accumulate_aggregate(aggregate_op::max, {}, integer(-9), true, 128);
  max      = accumulate_aggregate(aggregate_op::max, max, integer(7), true, 128);
  equal(finalize_aggregate(aggregate_op::max, max, {64, 9, 0}, {64, 9, 0}).value, integer(7));
}
TEST_CASE("MO equality keys normalize scales signs and zero without loss", "[decimal_aggregate]")
{
  for (auto sign : {-1, 1}) {
    auto a = normalize_key(integer(sign * 1200), 3), b = normalize_key(integer(sign * 12), 1);
    equal(a.value, b.value);
    CHECK(a.scale == b.scale);
    auto distinct = normalize_key(integer(sign * 1201), 3);
    CHECK(compare_coefficient(a.value, distinct.value) != 0);
  }
  auto zero = normalize_key({}, 76);
  equal(zero.value, {});
  CHECK(zero.scale == 0);
}
