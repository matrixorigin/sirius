/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "numeric/exact_decimal.hpp"

#include <catch.hpp>

#include <array>
#include <random>
#include <string>

using namespace sirius::mo_decimal;
#include "exact_decimal_wide_cases.hpp"
namespace {
// Test-only independent decimal-string fixture builder, not a production codec
// or an arithmetic oracle that calls the implementation under test.
coefficient number(std::string digits)
{
  coefficient result;
  bool negative = digits[0] == '-';
  for (auto digit : digits.substr(negative ? 1 : 0)) {
    uint64_t carry = digit - '0';
    for (auto& word : result.words) {
      auto next = uint64_t(word) * 10 + carry;
      word      = next;
      carry     = next >> 32;
    }
    REQUIRE(carry == 0);
  }
  if (negative) {
    uint64_t carry = 1;
    for (auto& word : result.words) {
      auto next = uint64_t(~word) + carry;
      word      = next;
      carry     = next >> 32;
    }
  }
  return result;
}
void expect(decimal_result const& result, std::string const& value)
{
  REQUIRE(result.error == decimal_error::none);
  REQUIRE(result.valid);
  auto expected = number(value);
  for (int i = 0; i < 8; ++i)
    CHECK(result.value.words[i] == expected.words[i]);
}
decimal_result run(decimal_op op,
                   std::string const& a,
                   decimal_type at,
                   std::string const& b,
                   decimal_type bt,
                   decimal_type out)
{
  return evaluate_decimal(op, number(a), at, true, number(b), bt, true, out);
}
}  // namespace

TEST_CASE("MO exact descriptor and coefficient bytes preserve physical width", "[exact_decimal]")
{
  CHECK((decimal_type{256, 15, 2}.valid()));
  CHECK_FALSE((decimal_type{128, 39, 0}.valid()));
  CHECK_FALSE((decimal_type{256, 77, 0}.valid()));
  CHECK_FALSE((decimal_type{64, 15, 16}.valid()));
  CHECK_FALSE((decimal_type{32, 9, 0}.valid()));
  for (auto width : {8u, 16u, 32u}) {
    for (auto n : {"0", "1", "-1", "9007199254740993", "-123456789"}) {
      auto original = number(n);
      std::array<uint8_t, 32> bytes{};
      store_coefficient(original, bytes.data(), width);
      auto observed = load_coefficient(bytes.data(), width);
      for (int i = 0; i < 8; ++i)
        CHECK(observed.words[i] == original.words[i]);
    }
  }
}

TEST_CASE("MO exact scalars widen and round at the declared boundary", "[exact_decimal]")
{
  auto max38 = std::string(38, '9');
  expect(run(decimal_op::add, max38, {128, 38, 0}, "1", {64, 1, 0}, {256, 39, 0}),
         "1" + std::string(38, '0'));
  expect(run(decimal_op::subtract, "-12", {64, 2, 0}, "5", {64, 1, 0}, {64, 3, 0}), "-17");
  expect(run(decimal_op::add, "100", {64, 3, 2}, "-2", {64, 1, 0}, {64, 3, 2}), "-100");
  for (auto sign : {std::string(), std::string("-")}) {
    expect(run(decimal_op::cast, sign + "125", {64, 3, 2}, "0", {64, 1, 0}, {64, 2, 1}),
           sign + "13");
    expect(run(decimal_op::divide, sign + "125", {64, 3, 2}, "1", {64, 1, 0}, {128, 38, 1}),
           sign + "13");
    expect(run(decimal_op::integer_divide, sign + "125", {64, 3, 2}, "1", {64, 1, 0}, {64, 18, 0}),
           sign + "1");
    expect(run(decimal_op::modulo, sign + "125", {64, 3, 2}, "10", {64, 2, 1}, {64, 3, 2}),
           sign + "25");
  }
  // Raw product 10^80 is wider than 256 bits; its once-scaled 10^50 result fits.
  expect(run(decimal_op::multiply,
             "1" + std::string(40, '0'),
             {256, 41, 30},
             "1" + std::string(40, '0'),
             {256, 41, 30},
             {256, 65, 30}),
         "1" + std::string(50, '0'));
  for (auto scale : {2u, 6u, 12u, 30u})
    expect(run(decimal_op::divide,
               "100",
               {64, 10, 2},
               "300",
               {64, 10, 2},
               {256, 65, static_cast<uint8_t>(scale)}),
           std::string(scale, '3'));
  expect(run(decimal_op::divide,
             "1" + std::string(36, '0'),
             {128, 38, 37},
             "2",
             {64, 1, 0},
             {128, 38, 30}),
         "5" + std::string(28, '0'));
}

TEST_CASE("MO exact errors NULLs masks and comparison remain distinct", "[exact_decimal]")
{
  auto maximum = std::string(65, '9');
  expect(run(decimal_op::add, maximum, {256, 65, 0}, "0", {64, 1, 0}, {256, 65, 0}), maximum);
  CHECK(run(decimal_op::add, maximum, {256, 65, 0}, "1", {64, 1, 0}, {256, 65, 0}).error ==
        decimal_error::out_of_range);
  CHECK(run(decimal_op::cast, "100", {64, 3, 0}, "0", {64, 1, 0}, {64, 2, 0}).error ==
        decimal_error::invalid_input);
  auto zero = number("0"), one = number("1");
  CHECK(evaluate_decimal(
          static_cast<decimal_op>(255), one, {64, 1, 0}, true, one, {64, 1, 0}, true, {64, 1, 0})
          .error == decimal_error::invalid_input);
  for (auto op : {decimal_op::divide, decimal_op::integer_divide, decimal_op::modulo}) {
    auto r = evaluate_decimal(op, one, {64, 1, 0}, true, zero, {64, 1, 0}, true, {64, 18, 0});
    CHECK_FALSE(r.valid);
    CHECK(r.error == decimal_error::none);
  }
  auto r = evaluate_decimal(decimal_op::add,
                            number(maximum),
                            {256, 65, 0},
                            true,
                            one,
                            {64, 1, 0},
                            true,
                            {256, 65, 0},
                            false);
  CHECK_FALSE(r.valid);
  CHECK(r.error == decimal_error::none);
  r = evaluate_decimal(decimal_op::multiply,
                       number(maximum),
                       {256, 65, 0},
                       false,
                       number(maximum),
                       {256, 65, 0},
                       true,
                       {256, 65, 0});
  CHECK_FALSE(r.valid);
  CHECK(r.error == decimal_error::none);
  expect(run(decimal_op::equal, "100", {64, 3, 2}, "10000", {128, 5, 4}, {64, 1, 0}), "1");
  expect(run(decimal_op::less,
             "-9007199254740993",
             {64, 18, 0},
             "-9007199254740992",
             {64, 18, 0},
             {64, 1, 0}),
         "1");
  expect(run(decimal_op::equal,
             "9007199254740993",
             {64, 18, 0},
             "9007199254740992",
             {64, 18, 0},
             {64, 1, 0}),
         "0");
  expect(run(decimal_op::integer_divide,
             "-9223372036854775808",
             {128, 38, 0},
             "1",
             {64, 1, 0},
             {64, 18, 0}),
         "-9223372036854775808");
  CHECK(
    run(
      decimal_op::integer_divide, "9223372036854775808", {128, 38, 0}, "1", {64, 1, 0}, {64, 18, 0})
      .error == decimal_error::out_of_range);
}

TEST_CASE("MO exact small-domain operations match independent integer arithmetic",
          "[exact_decimal]")
{
  std::mt19937 random(28968);
  for (int i = 0; i < 256; ++i) {
    int64_t a = int64_t(random() % 2000001) - 1000000;
    int64_t b = int64_t(random() % 2000000) - 1000000;
    if (!b) b = 1;
    auto av = std::to_string(a), bv = std::to_string(b);
    expect(run(decimal_op::add, av, {64, 18, 0}, bv, {64, 18, 0}, {128, 38, 0}),
           std::to_string(a + b));
    expect(run(decimal_op::subtract, av, {64, 18, 0}, bv, {64, 18, 0}, {128, 38, 0}),
           std::to_string(a - b));
    expect(run(decimal_op::multiply, av, {64, 18, 0}, bv, {64, 18, 0}, {128, 38, 0}),
           std::to_string(a * b));
    expect(run(decimal_op::integer_divide, av, {64, 18, 0}, bv, {64, 18, 0}, {64, 18, 0}),
           std::to_string(a / b));
    expect(run(decimal_op::modulo, av, {64, 18, 0}, bv, {64, 18, 0}, {128, 38, 0}),
           std::to_string(a % b));
  }
}

TEST_CASE("MO exact wide operations match an independent integer rational oracle",
          "[exact_decimal]")
{
  for (auto const& test : wide_cases) {
    INFO("op=" << static_cast<int>(test.op) << " a=" << test.a << " b=" << test.b);
    auto result =
      run(test.op, test.a, {256, 76, test.sa}, test.b, {256, 76, test.sb}, {256, 65, test.so});
    if (test.overflow) {
      CHECK(result.error == decimal_error::out_of_range);
      CHECK_FALSE(result.valid);
    } else
      expect(result, test.expected);
  }
}
