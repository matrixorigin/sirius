/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "decimal_import_fixture.hpp"
#include "expression/ast/utils.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "expression_evaluator/gpu_expression_translator_internal.hpp"
#include "numeric/decimal_error.hpp"
#include "numeric/exact_decimal_gpu.hpp"
#include "sirius_c.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>

using namespace sirius;
using namespace sirius::mo_decimal;
namespace {
std::vector<coefficient> download(cudf::column_view column,
                                  uint32_t bytes,
                                  rmm::cuda_stream_view stream)
{
  std::vector<coefficient> result(column.size());
  if (bytes == 32) {
    std::vector<uint64_t> limb(column.size());
    for (int i = 0; i < 4; ++i) {
      REQUIRE(cudaMemcpyAsync(limb.data(),
                              column.child(3 - i).data<uint64_t>(),
                              limb.size() * 8,
                              cudaMemcpyDeviceToHost,
                              stream.value()) == cudaSuccess);
      stream.synchronize();
      for (std::size_t row = 0; row < result.size(); ++row) {
        result[row].words[2 * i]     = limb[row];
        result[row].words[2 * i + 1] = limb[row] >> 32;
      }
    }
  } else {
    std::vector<uint8_t> data(result.size() * bytes);
    REQUIRE(
      cudaMemcpyAsync(
        data.data(), column.data<uint8_t>(), data.size(), cudaMemcpyDeviceToHost, stream.value()) ==
      cudaSuccess);
    stream.synchronize();
    for (std::size_t row = 0; row < result.size(); ++row) {
      if (bytes == 1)
        result[row].words[0] = data[row];
      else
        result[row] = load_coefficient(data.data() + row * bytes, bytes);
    }
  }
  return result;
}
void expect(cudf::column_view column,
            uint32_t bytes,
            std::vector<int64_t> const& expected,
            rmm::cuda_stream_view stream)
{
  REQUIRE(column.size() == static_cast<cudf::size_type>(expected.size()));
  REQUIRE(column.null_count() == 0);
  auto values = download(column, bytes, stream);
  for (std::size_t row = 0; row < values.size(); ++row) {
    auto reference = decimal_fixture::small(expected[row]);
    for (int word = 0; word < 8; ++word)
      CHECK(values[row].words[word] == reference.words[word]);
  }
}
std::unique_ptr<ast::node> constant(int64_t value, logical_type const& type)
{
  return std::make_unique<ast::node>(ast::constant{decimal_fixture::small(value), type});
}
std::unique_ptr<ast::node> reference(uint32_t index, logical_type const& type)
{
  return std::make_unique<ast::node>(ast::reference{index, type});
}
std::unique_ptr<ast::node> binary(decimal_op op,
                                  std::unique_ptr<ast::node> left,
                                  std::unique_ptr<ast::node> right,
                                  logical_type const& output)
{
  std::vector<std::unique_ptr<ast::node>> arguments;
  arguments.push_back(std::move(left));
  arguments.push_back(std::move(right));
  return std::make_unique<ast::node>(
    ast::function_call{function(op), std::move(arguments), output});
}
std::unique_ptr<cudf::column> values(decimal_type type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr,
                                     bool second_null     = false,
                                     int64_t second_value = 125)
{
  auto column = make_decimal_literal(type, decimal_fixture::small(999), true, 2, stream, mr);
  auto view   = column->mutable_view();
  if (type.bits != 256) {
    uint8_t coefficient[16]{};
    store_coefficient(decimal_fixture::small(second_value), coefficient, type.bytes());
    REQUIRE(cudaMemcpyAsync(view.data<uint8_t>() + type.bytes(),
                            coefficient,
                            type.bytes(),
                            cudaMemcpyHostToDevice,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
  } else {
    for (int i = 0; i < 4; ++i) {
      auto coefficient = decimal_fixture::small(second_value);
      auto index       = (3 - i) * 2;
      uint64_t value   = uint64_t(coefficient.words[index]) | uint64_t(coefficient.words[index + 1])
                                                              << 32;
      REQUIRE(
        cudaMemcpyAsync(
          view.child(i).data<uint64_t>() + 1, &value, 8, cudaMemcpyHostToDevice, stream.value()) ==
        cudaSuccess);
      stream.synchronize();
    }
  }
  if (second_null) {
    uint32_t mask = 1;
    REQUIRE(cudaMemcpyAsync(view.null_mask(), &mask, 4, cudaMemcpyHostToDevice, stream.value()) ==
            cudaSuccess);
    stream.synchronize();
    column->set_null_count(1);
  }
  return column;
}
}  // namespace

TEST_CASE("MO exact null predicates consume canonical validity in every evaluator strategy",
          "[decimal_import_gpu]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  for (uint16_t bits : {64, 128, 256}) {
    auto type      = logical_type::make_mo_decimal({bits, 15, 2}, true);
    auto all_valid = values(type.mo_decimal_type(), stream, mr);
    auto nullable  = values(type.mo_decimal_type(), stream, mr, true);
    auto empty     = make_decimal_literal(type.mo_decimal_type(), {}, true, 0, stream, mr);
    auto slice     = cudf::slice(nullable->view(), {1, 2}, stream);
    std::vector<cudf::column_view> inputs{
      all_valid->view(), nullable->view(), slice[0], empty->view()};
    std::vector<std::vector<int64_t>> nulls{{0, 0}, {0, 1}, {1}, {}};
    for (auto strategy : {expression_evaluator_strategy::MATERIALIZE,
                          expression_evaluator_strategy::AST_INTERPRET,
                          expression_evaluator_strategy::AST_JIT}) {
      for (bool is_null : {false, true}) {
        for (bool parent_not : {false, true}) {
          INFO("bits=" << bits << " strategy=" << int(strategy) << " is_null=" << is_null
                       << " parent_not=" << parent_not);
          auto op = is_null ? ast::unary_op::kind::op_is_null : ast::unary_op::kind::op_is_not_null;
          auto expression = std::make_unique<ast::node>(ast::unary_op{op, reference(0, type)});
          gpu_expression_translator translator(stream.view(), mr);
          CHECK_FALSE(translator.translate_expression(*expression).has_value());
          if (parent_not)
            expression = std::make_unique<ast::node>(
              ast::unary_op{ast::unary_op::kind::op_not, std::move(expression)});
          expression_evaluator evaluator(*expression, mr, stream.view(), strategy);
          for (std::size_t i = 0; i < inputs.size(); ++i) {
            auto expected = nulls[i];
            if (is_null == parent_not)
              for (auto& value : expected)
                value = !value;
            auto result = evaluator.evaluate(cudf::table_view{{inputs[i]}});
            CHECK(result->view().column(0).type().id() == cudf::type_id::BOOL8);
            expect(result->view().column(0), 1, expected, stream.view());
          }
        }
      }
    }
    for (bool is_null : {false, true}) {
      auto op = is_null ? ast::unary_op::kind::op_is_null : ast::unary_op::kind::op_is_not_null;
      ast::node expression(
        ast::unary_op{op, std::make_unique<ast::node>(ast::constant{sirius::null_value{}, type})});
      expression_evaluator evaluator(expression, mr, stream.view());
      auto result = evaluator.evaluate(cudf::table_view{{all_valid->view()}});
      expect(result->view().column(0), 1, {is_null, is_null}, stream.view());
    }
  }
}

TEST_CASE("MO imported scalar signatures execute on the GPU without coercion",
          "[decimal_import_gpu]")
{
  decimal_fixture::importer importer;
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  char const* names[]{"mo_decimal_add",
                      "mo_decimal_subtract",
                      "mo_decimal_multiply",
                      "mo_decimal_divide",
                      "mo_decimal_integer_divide",
                      "mo_decimal_modulo",
                      "mo_decimal_negate",
                      "mo_decimal_cast",
                      "mo_decimal_equal",
                      "mo_decimal_not_equal",
                      "mo_decimal_less",
                      "mo_decimal_less_equal",
                      "mo_decimal_greater",
                      "mo_decimal_greater_equal"};
  int64_t expected[]{225, 25, 125, 125, 1, 25, -125, 13, 0, 1, 0, 0, 1, 1};
  for (uint16_t bits : {64, 128, 256}) {
    decimal_type input{bits, 9, 2};
    auto dummy = make_decimal_literal(input, {}, true, 3, stream, mr);
    for (unsigned i = 0; i < 14; ++i) {
      INFO("bits=" << bits << " function=" << names[i]);
      auto op    = static_cast<decimal_op>(i);
      bool unary = op == decimal_op::cast || op == decimal_op::negate;
      decimal_type output{bits, 15, static_cast<uint8_t>(op == decimal_op::cast ? 1 : 2)};
      auto result_type = decimal_fixture::type(output);
      if (op >= decimal_op::equal)
        result_type.mutable_bool_()->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
      else if (op == decimal_op::integer_divide)
        result_type.mutable_i64()->set_nullability(substrait::Type::NULLABILITY_NULLABLE);
      std::vector<substrait::Expression> arguments{decimal_fixture::literal(input, 125)};
      if (!unary) arguments.push_back(decimal_fixture::literal(input, 100));
      auto plan       = decimal_fixture::plan(names[i], result_type, std::move(arguments));
      auto expression = importer.bind(plan);
      expression_evaluator evaluator(*expression, mr, stream.view());
      auto result = evaluator.evaluate(cudf::table_view{{dummy->view()}});
      auto bytes  = op >= decimal_op::equal            ? 1u
                    : op == decimal_op::integer_divide ? 8u
                                                       : input.bytes();
      expect(
        result->view().column(0), bytes, {expected[i], expected[i], expected[i]}, stream.view());
      if (op < decimal_op::equal && op != decimal_op::integer_divide)
        CHECK(decimal_column_matches(result->view().column(0), output));
    }
  }
  CHECK((sirius_capabilities() & SIRIUS_CAP_MO_EXACT_DECIMAL_V1) != 0);
}

TEST_CASE("MO imported literals preserve distinct high limbs and signs", "[decimal_import_gpu]")
{
  decimal_fixture::importer importer;
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  for (uint16_t bits : {64, 128, 256}) {
    decimal_type type{bits, static_cast<uint8_t>(bits == 64 ? 18 : bits == 128 ? 38 : 65), 4};
    coefficient value;
    value.words[0] = 0x89abcdef;
    value.words[1] = 0x01234567;
    if (bits >= 128) {
      value.words[2] = 0xaabbccdd;
      value.words[3] = 0x01234567;
    }
    if (bits == 256) {
      value.words[4] = 0x12345678;
      value.words[5] = 0x9abcdef0;
      value.words[6] = 5;
    }
    auto dummy = make_decimal_literal(type, {}, true, 2, stream, mr);
    for (bool negative : {false, true}) {
      auto expected = value;
      if (negative) {
        uint64_t carry = 1;
        for (auto& word : expected.words) {
          auto next = uint64_t(~word) + carry;
          word      = next;
          carry     = next >> 32;
        }
      }
      auto plan = decimal_fixture::plan(
        "mo_decimal_cast", decimal_fixture::type(type), {decimal_fixture::literal(type, expected)});
      auto expression = importer.bind(plan);
      expression_evaluator evaluator(*expression, mr, stream.view());
      auto output   = evaluator.evaluate(cudf::table_view{{dummy->view()}});
      auto observed = download(output->view().column(0), type.bytes(), stream.view());
      REQUIRE(output->view().column(0).null_count() == 0);
      for (auto const& row : observed)
        for (int word = 0; word < 8; ++word)
          CHECK(row.words[word] == expected.words[word]);
    }
  }
}

TEST_CASE("MO integer casts preserve the full signed source domain", "[decimal_import_gpu]")
{
  rmm::cuda_stream stream;
  auto mr    = cudf::get_current_device_resource_ref();
  auto input = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
  int64_t endpoints[]{INT64_MIN, INT64_MAX};
  REQUIRE(cudaMemcpyAsync(input->mutable_view().data<int64_t>(),
                          endpoints,
                          sizeof(endpoints),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  auto type = logical_type::make_mo_decimal({128, 19, 0}, false);
  std::vector<std::unique_ptr<ast::node>> arguments;
  arguments.push_back(reference(0, logical_type::make(type_id::BIGINT)));
  ast::node cast(ast::function_call{function(decimal_op::cast), std::move(arguments), type});
  expression_evaluator evaluator(cast, mr, stream.view());
  auto output = evaluator.evaluate(cudf::table_view{{input->view()}});
  expect(output->view().column(0), 16, {INT64_MIN, INT64_MAX}, stream.view());
}

TEST_CASE("MO conditional masks suppress inactive arithmetic and cast errors",
          "[decimal_import_gpu]")
{
  rmm::cuda_stream stream;
  auto mr = cudf::get_current_device_resource_ref();
  for (uint16_t bits : {64, 128, 256}) {
    INFO("conditional bits=" << bits);
    auto type           = logical_type::make_mo_decimal({bits, 3, 0}, false);
    auto input          = values(type.mo_decimal_type(), stream, mr);
    auto nullable_input = values(type.mo_decimal_type(), stream, mr, true);
    auto predicate      = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
    bool selected[]{false, true};
    REQUIRE(cudaMemcpyAsync(predicate->mutable_view().data<bool>(),
                            selected,
                            sizeof(selected),
                            cudaMemcpyHostToDevice,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    std::vector<ast::case_expr::when_then> branches;
    branches.push_back({reference(1, logical_type::make(type_id::BOOLEAN)),
                        binary(decimal_op::add, reference(0, type), constant(1, type), type)});
    auto expression =
      std::make_unique<ast::node>(ast::case_expr{std::move(branches), constant(9, type), type});
    expression_evaluator case_only(*expression, mr, stream.view());
    auto case_result = case_only.evaluate(cudf::table_view{{input->view(), predicate->view()}});
    INFO("CASE parent nullable=" << case_result->view().column(0).nullable());
    if (bits == 256) {
      INFO("CASE high limb nullable=" << case_result->view().column(0).child(0).nullable());
      REQUIRE(decimal_column_matches(case_result->view().column(0), type.mo_decimal_type()));
    }
    // A following exact operation also checks that conditional output retained
    // the canonical parent-validity-only Decimal256 representation.
    expression = binary(decimal_op::add, std::move(expression), constant(1, type), type);
    expression_evaluator evaluator(*expression, mr, stream.view());
    auto result = evaluator.evaluate(cudf::table_view{{input->view(), predicate->view()}});
    expect(result->view().column(0), type.fixed_width_byte_size(), {10, 127}, stream.view());
    CHECK(decimal_column_matches(result->view().column(0), type.mo_decimal_type()));
    selected[0] = true;
    REQUIRE(cudaMemcpyAsync(predicate->mutable_view().data<bool>(),
                            selected,
                            sizeof(selected),
                            cudaMemcpyHostToDevice,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    try {
      evaluator.evaluate(cudf::table_view{{input->view(), predicate->view()}});
      FAIL("active CASE overflow was ignored");
    } catch (numeric_error const& error) {
      CHECK(error.code() == decimal_error::out_of_range);
    }
    selected[0] = false;
    REQUIRE(cudaMemcpyAsync(predicate->mutable_view().data<bool>(),
                            selected,
                            sizeof(selected),
                            cudaMemcpyHostToDevice,
                            stream.value()) == cudaSuccess);
    stream.synchronize();
    auto recovered = evaluator.evaluate(cudf::table_view{{input->view(), predicate->view()}});
    expect(recovered->view().column(0), type.fixed_width_byte_size(), {10, 127}, stream.view());

    auto narrow_type = logical_type::make_mo_decimal({bits, 2, 0}, false);
    auto cast_input  = values(type.mo_decimal_type(), stream, mr, false, 12);
    std::vector<std::unique_ptr<ast::node>> masked_cast_arguments;
    masked_cast_arguments.push_back(reference(0, type));
    auto masked_cast = std::make_unique<ast::node>(ast::function_call{
      function(decimal_op::cast), std::move(masked_cast_arguments), narrow_type});
    std::vector<ast::case_expr::when_then> cast_branches;
    cast_branches.push_back(
      {reference(1, logical_type::make(type_id::BOOLEAN)), std::move(masked_cast)});
    ast::node cast_case(
      ast::case_expr{std::move(cast_branches), constant(9, narrow_type), narrow_type});
    expression_evaluator cast_selector(cast_case, mr, stream.view());
    auto cast_result =
      cast_selector.evaluate(cudf::table_view{{cast_input->view(), predicate->view()}});
    expect(cast_result->view().column(0), type.fixed_width_byte_size(), {9, 12}, stream.view());

    auto nullable_type = logical_type::make_mo_decimal({bits, 3, 0}, true);
    std::vector<std::unique_ptr<ast::node>> alternatives;
    alternatives.push_back(reference(1, nullable_type));
    alternatives.push_back(binary(decimal_op::add, reference(0, type), constant(1, type), type));
    ast::node coalesce(ast::coalesce{std::move(alternatives), nullable_type});
    expression_evaluator coalescer(coalesce, mr, stream.view());
    auto combined = coalescer.evaluate(cudf::table_view{{input->view(), nullable_input->view()}});
    expect(combined->view().column(0), type.fixed_width_byte_size(), {999, 126}, stream.view());
    auto active = binary(decimal_op::add, reference(0, type), constant(1, type), type);
    expression_evaluator failing(*active, mr, stream.view());
    try {
      failing.evaluate(cudf::table_view{{input->view()}});
      FAIL("active arithmetic overflow was ignored");
    } catch (numeric_error const& error) {
      CHECK(error.code() == decimal_error::out_of_range);
    }
    std::vector<std::unique_ptr<ast::node>> cast_arguments;
    cast_arguments.push_back(reference(0, type));
    ast::node cast(ast::function_call{function(decimal_op::cast),
                                      std::move(cast_arguments),
                                      logical_type::make_mo_decimal({bits, 2, 0}, false)});
    expression_evaluator narrowing(cast, mr, stream.view());
    try {
      narrowing.evaluate(cudf::table_view{{input->view()}});
      FAIL("active cast overflow was ignored");
    } catch (numeric_error const& error) {
      CHECK(error.code() == decimal_error::invalid_input);
    }
  }
}

TEST_CASE("MO SELECT zero divisors produce NULL without numeric failure", "[decimal_import_gpu]")
{
  rmm::cuda_stream stream;
  auto mr          = cudf::get_current_device_resource_ref();
  auto input_type  = logical_type::make_mo_decimal({64, 9, 0}, false);
  auto output_type = logical_type::make_mo_decimal({128, 19, 2}, true);
  auto dummy       = make_decimal_literal(input_type.mo_decimal_type(), {}, true, 2, stream, mr);
  for (int64_t divisor : {0, 2}) {
    auto expression = binary(
      decimal_op::divide, constant(1, input_type), constant(divisor, input_type), output_type);
    expression_evaluator evaluator(*expression, mr, stream.view());
    auto result = evaluator.evaluate(cudf::table_view{{dummy->view()}});
    if (divisor == 0)
      CHECK(result->view().column(0).null_count() == 2);
    else
      expect(result->view().column(0), 16, {50, 50}, stream.view());
  }
}

TEST_CASE("MO references reject width restoration while ordinary decimals retain it",
          "[decimal_import_gpu]")
{
  rmm::cuda_stream stream;
  auto mr         = cudf::get_current_device_resource_ref();
  auto compressed = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::DECIMAL32, 0}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemsetAsync(compressed->mutable_view().data<uint8_t>(), 0, 8, stream.value()) ==
          cudaSuccess);
  stream.synchronize();
  auto exact = reference(0, logical_type::make_mo_decimal({64, 18, 0}, false));
  expression_evaluator strict(*exact, mr, stream.view());
  REQUIRE_THROWS_AS(strict.evaluate(cudf::table_view{{compressed->view()}}), std::invalid_argument);
  auto ordinary = reference(0, logical_type::make_decimal(18, 0));
  expression_evaluator restoring(*ordinary, mr, stream.view());
  auto output = restoring.evaluate(cudf::table_view{{compressed->view()}});
  CHECK(output->view().column(0).type() == cudf::data_type{cudf::type_id::DECIMAL64, 0});
  expect(output->view().column(0), 8, {0, 0}, stream.view());
}

TEST_CASE("MO exact GPU expressions retain empty schemas and evaluator reuse",
          "[decimal_import_gpu]")
{
  rmm::cuda_stream stream;
  auto mr         = cudf::get_current_device_resource_ref();
  auto type       = logical_type::make_mo_decimal({256, 15, 2}, false);
  auto expression = binary(decimal_op::add, constant(125, type), constant(-25, type), type);
  expression_evaluator evaluator(*expression, mr, stream.view());
  for (int rows : {0, 2, 0}) {
    auto input  = make_decimal_literal(type.mo_decimal_type(), {}, true, rows, stream, mr);
    auto output = evaluator.evaluate(cudf::table_view{{input->view()}});
    CHECK(output->num_rows() == rows);
    CHECK(decimal_column_matches(output->view().column(0), type.mo_decimal_type()));
    if (rows) expect(output->view().column(0), 32, {100, 100}, stream.view());
  }
}
